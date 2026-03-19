from models.model_loader import ModelLoader
from app.config import TOKEN_LIMITS


class Translator:

    def __init__(self):
        self.model = ModelLoader()

    def translate(self, text, target_language):

        prompt = f"""[INST] Translate the following text to {target_language}. Output only the translated text, nothing else. Do not add explanations or examples.

        Text: {text} [/INST]"""

        return self.model.generate(prompt, TOKEN_LIMITS["translation"])