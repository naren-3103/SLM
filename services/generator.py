from models.model_loader import ModelLoader
from app.config import TOKEN_LIMITS


class TextGenerator:

    def __init__(self):
        self.model = ModelLoader()

    def generate(self, prompt):

        prompt = f"""[INST] {prompt} [/INST]"""

        return self.model.generate(prompt, TOKEN_LIMITS["generation"])