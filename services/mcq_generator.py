from models.model_loader import ModelLoader
from rag_pipeline.embedding import Embedder
from rag_pipeline.index_documents import build_vector_db
from app.config import TOKEN_LIMITS, TOP_K, SCORE_THRESHOLD


class MCQGenerator:

    def __init__(self):
        self.model = ModelLoader()
        self.embedder = Embedder()
        self.vector_db = build_vector_db()

    def generate_questions(self, topic, count, category):

        # 1. Fetch context from Vector DB
        context = "No relevant context found."
        if self.vector_db:
            query_embedding = self.embedder.encode_query(topic)
            results = self.vector_db.search_hybrid(
                query_text=topic,
                query_embedding=query_embedding,
                k=TOP_K * 2,  # Fetch more chunks to have enough context for multiple questions
                score_threshold=SCORE_THRESHOLD,
            )
            if results:
                context = "\n\n".join([r['text'] for r in results])

        # 2. Build prompt
        prompt = f"""[INST] You are an expert quiz master. Use ONLY the provided context to generate exactly {count} multiple-choice question(s) of '{category}' difficulty level about '{topic}'.
If the context does not contain enough information, generate as many questions as you can based on the context.
Each question must have exactly 4 options and clearly indicate the correct answer.
Format each question clearly with the question text, the 4 options labeled A, B, C, D, and the correct answer at the end.

Context:
{context}
[/INST]"""

        stop_tokens = ["</s>", "[INST]", "[/INST]"]
        return self.model.generate(prompt, TOKEN_LIMITS.get("mcq", 2000), stop=stop_tokens)
