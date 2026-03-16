from models.model_loader import ModelLoader
from app.config import TOKEN_LIMITS


class TextSummarizer:

    def __init__(self):
        self.model = ModelLoader()

    def summarize_text(self, text):

        prompt = f"""
Summarize the following text.

{text}

Summary:
"""

        return self.model.generate(
            prompt,
            TOKEN_LIMITS["summarization"]
        )