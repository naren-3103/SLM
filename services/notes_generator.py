from models.model_loader import ModelLoader
from rag_pipeline.embedding import Embedder
from rag_pipeline.index_documents import build_vector_db
from app.config import TOKEN_LIMITS, TOP_K, SCORE_THRESHOLD


class NotesGenerator:

    def __init__(self):
        self.model = ModelLoader()
        self.embedder = Embedder()
        self.vector_db = build_vector_db()

    def generate_notes(self, topic):

        # 1. Fetch context from Vector DB
        context = "No relevant context found."
        if self.vector_db:
            query_embedding = self.embedder.encode_query(topic)
            results = self.vector_db.search_hybrid(
                query_text=topic,
                query_embedding=query_embedding,
                k=TOP_K * 2,  # Fetch ample chunks for comprehensive notes
                score_threshold=SCORE_THRESHOLD,
            )
            if results:
                context = "\n\n".join([r['text'] for r in results])

        # 2. Build prompt
        prompt = f"""[INST] You are an expert educator creating study materials. Use ONLY the provided context to generate comprehensive, student-friendly notes about '{topic}'.
Ensure the notes are engaging and easy to understand.
Format the notes strictly using:
- Main Headers and Subheaders (using Markdown # and ##)
- Clear Pointers and Bulleted lists
- Include Examples wherever possible based on the context to illustrate the points.
Do not include information outside of the given context. If the context does lack information, do the best with what is provided.

Context:
{context}
[/INST]"""

        stop_tokens = ["</s>", "[INST]", "[/INST]"]
        return self.model.generate(prompt, TOKEN_LIMITS.get("notes", 2000), stop=stop_tokens)
