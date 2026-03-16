from rag_pipeline.document_loader import load_documents
from rag_pipeline.chunking import chunk_text
from rag_pipeline.embedding import Embedder
from rag_pipeline.vector_store import VectorStore
from app.config import DOCUMENT_PATH


def build_vector_db():

    print("Loading documents...")

    documents = load_documents(DOCUMENT_PATH)

    if len(documents) == 0:
        print("No documents found!")
        return None

    all_chunks = []

    for doc in documents:

        chunks = chunk_text(doc)

        all_chunks.extend(chunks)

    print("Total chunks created:", len(all_chunks))

    embedder = Embedder()

    embeddings = embedder.encode(all_chunks)

    vector_db = VectorStore(len(embeddings[0]))

    vector_db.add(embeddings, all_chunks)

    print("Vector database created.")

    return vector_db