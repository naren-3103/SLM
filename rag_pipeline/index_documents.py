"""
Build (or reload) the vector database from documents in DOCUMENT_PATH.

Flow
----
1. Try to load an existing persisted index from VECTOR_DB_PATH.
2. If none found, load documents → chunk → embed → index → save.
"""

from rag_pipeline.document_loader import load_documents
from rag_pipeline.chunking import chunk_documents
from rag_pipeline.embedding import Embedder
from rag_pipeline.vector_store import VectorStore
from app.config import DOCUMENT_PATH, VECTOR_DB_PATH


def build_vector_db(force_rebuild: bool = False) -> VectorStore | None:
    """
    Return a ready-to-use VectorStore.

    Parameters
    ----------
    force_rebuild : bool
        Skip loading a cached index and always rebuild from documents.
    """
    # ── 1. Try loading persisted index ───────────────────────────────────────
    if not force_rebuild:
        store = VectorStore.load(VECTOR_DB_PATH)
        if store is not None:
            print(f"[IndexDocuments] Using cached index ({len(store)} chunks).")
            return store

    # ── 2. Load documents ────────────────────────────────────────────────────
    print("[IndexDocuments] Loading documents...")
    pages = load_documents(DOCUMENT_PATH)

    if not pages:
        print("[IndexDocuments] No documents found — vector DB not created.")
        return None

    # ── 3. Chunk ─────────────────────────────────────────────────────────────
    print("[IndexDocuments] Chunking documents...")
    chunks = chunk_documents(pages)
    print(f"[IndexDocuments] {len(chunks)} unique chunks created.")

    if not chunks:
        print("[IndexDocuments] All chunks were duplicates — nothing to index.")
        return None

    # ── 4. Embed ─────────────────────────────────────────────────────────────
    print("[IndexDocuments] Embedding chunks...")
    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts)

    # ── 5. Index ─────────────────────────────────────────────────────────────
    dim = embeddings.shape[1]
    vector_db = VectorStore(dim)
    vector_db.add(embeddings, chunks)

    # ── 6. Persist ───────────────────────────────────────────────────────────
    vector_db.save(VECTOR_DB_PATH)
    print("[IndexDocuments] Vector database built and saved.")

    return vector_db