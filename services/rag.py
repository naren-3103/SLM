"""
RAG service — wires together the embedding, vector store, and language model.

Changes vs. original
---------------------
* Uses ``Embedder.encode_query`` for single-query encoding.
* Calls ``VectorStore.search_hybrid`` (BM25 + vector + MMR).
* Filters results below SCORE_THRESHOLD.
* Embeds source citations into the context block.
* Returns a structured dict ``{answer, sources}`` instead of a plain string.
"""

from models.model_loader import ModelLoader
from rag_pipeline.embedding import Embedder
from rag_pipeline.index_documents import build_vector_db
from app.config import TOKEN_LIMITS, TOP_K, SCORE_THRESHOLD


class RAGPipeline:

    def __init__(self, force_rebuild: bool = False):
        self.model    = ModelLoader()
        self.embedder = Embedder()

        print("Building / loading vector database...")
        self.vector_db = build_vector_db(force_rebuild=force_rebuild)

        if self.vector_db is None:
            raise ValueError(
                "Vector database could not be created. "
                "Make sure documents exist in the configured DOCUMENT_PATH."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict:
        """
        Answer *question* using retrieved context.

        Returns
        -------
        dict with keys:
            "answer"  – str, generated answer from the LLM
            "sources" – list of dicts [{text, source, page, score}]
        """
        # 1. Encode query
        query_embedding = self.embedder.encode_query(question)

        # 2. Hybrid retrieval + MMR reranking
        results = self.vector_db.search_hybrid(
            query_text=question,
            query_embedding=query_embedding,
            k=TOP_K,
            score_threshold=SCORE_THRESHOLD,
        )

        if not results:
            return {
                "answer":  "I could not find any relevant information in the documents.",
                "sources": [],
            }

        # 3. Build context block with inline source citations
        context_parts = []
        for i, r in enumerate(results, start=1):
            citation = f"[{i}] ({r['source']}, page {r['page']})"
            context_parts.append(f"{citation}\n{r['text']}")
        context = "\n\n".join(context_parts)

        # 4. Prompt  (Mistral-instruct format)
        prompt = (
            "[INST] You are a research assistant. "
            "Answer the question using ONLY the provided context. "
            "Reference the source numbers (e.g., [1], [2]) when citing information. "
            "If the answer is not in the context, say \"Not found in the document\".\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question} [/INST]"
        )

        # 5. Generate
        answer = self.model.generate(prompt, TOKEN_LIMITS["rag"])

        return {
            "answer":  answer,
            "sources": results,
        }