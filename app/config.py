import os
MODEL_NAME = "microsoft/phi-2"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_PATH = "models/phi2"

VECTOR_DB_PATH = "data/vector_db"
DOCUMENT_PATH = "data/documents"

# --- Chunking ---
CHUNK_SIZE = 512          # Max characters per chunk
CHUNK_OVERLAP = 80        # Characters of overlap between adjacent chunks

# --- Retrieval ---
TOP_K = 5                 # Number of candidates to retrieve from vector store
MMR_LAMBDA = 0.6          # MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity
SCORE_THRESHOLD = 0.0     # Minimum similarity score to include a result (0.0 = disabled)
HYBRID_ALPHA = 0.7        # Weight for vector score vs BM25 (1.0 = pure vector, 0.0 = pure BM25)

# Generation parameters
GENERATION_CONFIG = {
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 40,
    "do_sample": False
}

# Token limits for each task
TOKEN_LIMITS = {
    "translation": 40,
    "summarization": 120,
    "rag": 200,
    "generation": 200
}