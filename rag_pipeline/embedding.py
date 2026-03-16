from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL

class Embedder:

    def __init__(self):

        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def encode(self, texts):

        return self.model.encode(texts)