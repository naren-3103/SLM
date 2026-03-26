"""
Advanced VectorStore backed by FAISS HNSW index.

Features
--------
- HNSW approximate nearest-neighbour search (fast on 1k+ chunks)
- Metadata store (source, page, chunk_id per chunk)
- Hybrid BM25 + vector retrieval
- MMR (Maximal Marginal Relevance) reranking for diversity
- save() / load() for index persistence
"""

import os
import pickle

import faiss
import numpy as np

from app.config import MMR_LAMBDA, HYBRID_ALPHA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mmr(
    query_vec: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_indices: list[int],
    scores: list[float],
    k: int,
    lambda_: float,
) -> list[int]:
    """
    Maximal Marginal Relevance selection.

    Parameters
    ----------
    query_vec           : 1-D query embedding (normalised)
    candidate_embeddings: 2-D array of candidate embeddings (normalised)
    candidate_indices   : positional indices into the full store
    scores              : similarity scores (higher = more relevant)
    k                   : number of results to return
    lambda_             : relevance weight  (1.0 = pure relevance, 0.0 = pure diversity)
    """
    selected: list[int] = []
    remaining = list(range(len(candidate_indices)))

    for _ in range(min(k, len(remaining))):
        if not remaining:
            break

        if not selected:
            # First pick: highest relevance score
            best_local = max(remaining, key=lambda i: scores[i])
        else:
            # MMR: relevance - lambda * max_sim_to_selected
            selected_embs = candidate_embeddings[[candidate_indices[s] for s in selected]]

            def mmr_score(i):
                rel = scores[i]
                sim_to_selected = float(np.max(candidate_embeddings[candidate_indices[i]] @ selected_embs.T))
                return lambda_ * rel - (1 - lambda_) * sim_to_selected

            best_local = max(remaining, key=mmr_score)

        selected.append(best_local)
        remaining.remove(best_local)

    return [candidate_indices[i] for i in selected]


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    HNSW-backed vector store with hybrid retrieval and MMR reranking.
    """

    # HNSW construction parameters
    _HNSW_M = 32           # edges per node  (higher → better recall, more RAM)
    _HNSW_EF_CONSTR = 200  # construction-time search depth

    def __init__(self, dim: int):
        self.dim = dim
        # Inner-product index works as cosine similarity when vectors are L2-normalised
        self.index = faiss.IndexHNSWFlat(dim, self._HNSW_M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self._HNSW_EF_CONSTR
        self.index.hnsw.efSearch = 64    # runtime search depth

        # Parallel metadata store
        self.texts: list[str] = []
        self.metadata: list[dict] = []   # [{source, page, chunk_id}]
        self._bm25 = None                # lazy-built on first search

    # ------------------------------------------------------------------
    # Mutating
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray, chunks: list[dict]):
        """
        Add chunk dicts (with 'text', 'source', 'page', 'chunk_id') and
        their corresponding embeddings.
        """
        vecs = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vecs)         # ensure unit norm
        self.index.add(vecs)

        for chunk in chunks:
            self.texts.append(chunk["text"])
            self.metadata.append({
                "source":   chunk.get("source", "unknown"),
                "page":     chunk.get("page", 1),
                "chunk_id": chunk.get("chunk_id", ""),
            })

        self._bm25 = None                # invalidate cached BM25 model

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _build_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [t.lower().split() for t in self.texts]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            self._bm25 = None

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        mmr_lambda: float = MMR_LAMBDA,
        hybrid_alpha: float = HYBRID_ALPHA,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Retrieve the top-k most relevant, diverse chunks.

        Returns
        -------
        list of dicts: [{text, source, page, chunk_id, score}]
        """
        if not self.texts:
            return []

        # Oversample candidates for MMR (fetch 3× the requested k)
        n_candidates = min(k * 3, len(self.texts))

        # -- Vector search ------------------------------------------------
        qvec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(qvec)
        D, I = self.index.search(qvec, n_candidates)
        vec_scores_raw = D[0]    # inner-product == cosine similarity (normalised)
        candidate_ids  = I[0]

        valid_mask = candidate_ids >= 0
        candidate_ids  = candidate_ids[valid_mask]
        vec_scores_raw = vec_scores_raw[valid_mask]

        if len(candidate_ids) == 0:
            return []

        # Normalise vector scores to [0, 1]
        vmin, vmax = vec_scores_raw.min(), vec_scores_raw.max()
        if vmax > vmin:
            vec_scores = (vec_scores_raw - vmin) / (vmax - vmin)
        else:
            vec_scores = np.ones_like(vec_scores_raw)

        # -- BM25 (keyword) search ----------------------------------------
        bm25_scores = np.zeros(len(candidate_ids), dtype=np.float32)
        query_words = query_embedding  # placeholder; we need query text for BM25

        # We build BM25 on demand; use stored texts for candidate scoring
        if self._bm25 is None:
            self._build_bm25()

        # Note: BM25 needs the raw query string, not its embedding.
        # We skip BM25 blending here if it isn't available via the caller;
        # see HybridSearch wrapper below for full usage.
        # bm25_scores stay 0 if not provided → falls back to pure vector.

        # -- Hybrid blend -------------------------------------------------
        hybrid_scores = hybrid_alpha * vec_scores + (1 - hybrid_alpha) * bm25_scores

        # -- Score threshold filter ---------------------------------------
        keep_mask = hybrid_scores >= score_threshold
        candidate_ids  = candidate_ids[keep_mask]
        hybrid_scores  = hybrid_scores[keep_mask]

        if len(candidate_ids) == 0:
            return []

        # -- MMR reranking ------------------------------------------------
        cand_embeddings = self._get_embeddings(candidate_ids)
        local_ids = list(range(len(candidate_ids)))
        mmr_order = _mmr(
            query_vec=qvec[0],
            candidate_embeddings=cand_embeddings,
            candidate_indices=local_ids,
            scores=hybrid_scores.tolist(),
            k=k,
            lambda_=mmr_lambda,
        )

        results = []
        for local_i in mmr_order:
            global_i = int(candidate_ids[local_i])
            results.append({
                "text":     self.texts[global_i],
                "source":   self.metadata[global_i]["source"],
                "page":     self.metadata[global_i]["page"],
                "chunk_id": self.metadata[global_i]["chunk_id"],
                "score":    float(hybrid_scores[local_i]),
            })

        return results

    def search_hybrid(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        k: int = 5,
        mmr_lambda: float = MMR_LAMBDA,
        hybrid_alpha: float = HYBRID_ALPHA,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Full hybrid search: combines vector similarity with BM25 keyword scores.
        Prefer this when you have the raw query string available.
        """
        if not self.texts:
            return []

        n_candidates = min(k * 3, len(self.texts))

        # Vector search
        qvec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(qvec)
        D, I = self.index.search(qvec, n_candidates)
        vec_scores_raw = D[0]
        candidate_ids  = I[0]

        valid_mask     = candidate_ids >= 0
        candidate_ids  = candidate_ids[valid_mask]
        vec_scores_raw = vec_scores_raw[valid_mask]

        if len(candidate_ids) == 0:
            return []

        vmin, vmax = vec_scores_raw.min(), vec_scores_raw.max()
        vec_scores = (vec_scores_raw - vmin) / (vmax - vmin) if vmax > vmin else np.ones_like(vec_scores_raw)

        # BM25 scores for every candidate
        if self._bm25 is None:
            self._build_bm25()

        bm25_scores = np.zeros(len(candidate_ids), dtype=np.float32)
        if self._bm25 is not None:
            query_tokens = query_text.lower().split()
            all_bm25 = np.array(self._bm25.get_scores(query_tokens), dtype=np.float32)
            raw_cand_bm25 = all_bm25[candidate_ids]
            bmin, bmax = raw_cand_bm25.min(), raw_cand_bm25.max()
            bm25_scores = (raw_cand_bm25 - bmin) / (bmax - bmin) if bmax > bmin else np.zeros_like(raw_cand_bm25)

        hybrid_scores = hybrid_alpha * vec_scores + (1 - hybrid_alpha) * bm25_scores
        keep_mask     = hybrid_scores >= score_threshold
        candidate_ids = candidate_ids[keep_mask]
        hybrid_scores = hybrid_scores[keep_mask]

        if len(candidate_ids) == 0:
            return []

        cand_embeddings = self._get_embeddings(candidate_ids)
        local_ids   = list(range(len(candidate_ids)))
        mmr_order   = _mmr(qvec[0], cand_embeddings, local_ids, hybrid_scores.tolist(), k, mmr_lambda)

        results = []
        for local_i in mmr_order:
            global_i = int(candidate_ids[local_i])
            results.append({
                "text":     self.texts[global_i],
                "source":   self.metadata[global_i]["source"],
                "page":     self.metadata[global_i]["page"],
                "chunk_id": self.metadata[global_i]["chunk_id"],
                "score":    float(hybrid_scores[local_i]),
            })

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str):
        """Persist the FAISS index and metadata to *directory*."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "metadata.pkl"), "wb") as fh:
            pickle.dump({"texts": self.texts, "metadata": self.metadata}, fh)
        print(f"[VectorStore] Saved {len(self.texts)} chunks to '{directory}'")

    @classmethod
    def load(cls, directory: str) -> "VectorStore | None":
        """Load a previously saved VectorStore. Returns None if not found."""
        index_path = os.path.join(directory, "index.faiss")
        meta_path  = os.path.join(directory, "metadata.pkl")
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            return None

        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as fh:
            payload = pickle.load(fh)

        store = cls.__new__(cls)
        store.dim      = index.d
        store.index    = index
        store.texts    = payload["texts"]
        store.metadata = payload["metadata"]
        store._bm25    = None
        print(f"[VectorStore] Loaded {len(store.texts)} chunks from '{directory}'")
        return store

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_embeddings(self, indices: np.ndarray) -> np.ndarray:
        """Reconstruct stored vectors from the HNSW index."""
        n = len(indices)
        vecs = np.empty((n, self.dim), dtype=np.float32)
        for i, idx in enumerate(indices):
            self.index.reconstruct(int(idx), vecs[i])
        return vecs

    def __len__(self) -> int:
        return len(self.texts)