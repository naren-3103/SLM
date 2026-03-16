from models.model_loader import ModelLoader
from app.config import TOKEN_LIMITS


class TextGenerator:

    def __init__(self):
        self.model = ModelLoader()

    def generate(self, prompt):

        return self.model.generate(
            prompt,
            TOKEN_LIMITS["generation"]
        )