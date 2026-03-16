from models.model_loader import ModelLoader
from app.config import TOKEN_LIMITS


class Translator:

    def __init__(self):
        self.model = ModelLoader()

    def translate(self, text, target_language):

        prompt = f"""
Translate the following text into {target_language}.

Text: {text}

Translation:
"""

        return self.model.generate(
            prompt,
            TOKEN_LIMITS["translation"]
        )