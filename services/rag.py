from models.model_loader import ModelLoader
from rag_pipeline.embedding import Embedder
from rag_pipeline.index_documents import build_vector_db
from app.config import TOKEN_LIMITS


class RAGPipeline:

    def __init__(self):

        self.model = ModelLoader()

        self.embedder = Embedder()

        print("Building vector database...")

        self.vector_db = build_vector_db()

        if self.vector_db is None:
            raise ValueError("Vector database was not created.")


    def ask(self, question):

        query_embedding = self.embedder.encode([question])[0]

        contexts = self.vector_db.search(query_embedding)

        context = " ".join(contexts)

        # Mistral Instruct format with strict grounding instruction
        prompt = f"""[INST] You are a research assistant. Answer the question using ONLY the provided context. If the answer is not present in the context, say "Not found in the document". Do not add any information from outside the context.

Context:
{context}

Question: {question} [/INST]"""

        answer = self.model.generate(
            prompt,
            TOKEN_LIMITS["rag"]
        )

        return answer