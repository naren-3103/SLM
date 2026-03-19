from models.model_loader import ModelLoader
from app.config import TOKEN_LIMITS


class TextSummarizer:

    def __init__(self):
        self.model = ModelLoader()

    def summarize_text(self, text):
        prompt = f"""[INST] Summarize the following text in a few concise sentences. Output only the summary, nothing else.

    Text: {text} [/INST]"""

        return self.model.generate(prompt, TOKEN_LIMITS["summarization"])