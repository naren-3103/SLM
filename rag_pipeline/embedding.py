"""
Embedder with normalized output, configurable batching, and a dedicated
query-encoding method.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL


class Embedder:

    def __init__(self, batch_size: int = 64):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into L2-normalised embeddings.
        Uses batching to avoid OOM on large corpora.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # cosine-similarity friendly
            show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string — returns a 1-D float32 array."""
        return self.encode([query])[0]