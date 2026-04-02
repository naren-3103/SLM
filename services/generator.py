"""
Service for generating raw text responses from the language model.
"""
from models.model_loader import ModelLoader
from app.config import TOKEN_LIMITS


class TextGenerator:
    """
    Handles prompt creation and model interaction for general text generation tasks.
    """

    def __init__(self):
        """Initializes the text generator with the singleton ModelLoader."""
        self.model = ModelLoader()

    def generate(self, prompt):
        """
        Generates text based on the provided prompt string.

        Args:
            prompt (str): The input text to serve as the prompt.
            
        Returns:
            str: The generated text response.
        """

        prompt = f"""[INST] {prompt} [/INST]"""

        return self.model.generate(prompt, TOKEN_LIMITS["generation"])