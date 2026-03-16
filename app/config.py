MODEL_NAME = "microsoft/phi-2"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MODEL_PATH = "models/phi2"

VECTOR_DB_PATH = "data/vector_db"
DOCUMENT_PATH = "data/documents"

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
    "rag": 150,
    "generation": 200
}