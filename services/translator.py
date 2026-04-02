"""
Service for translating text into a target language.
"""
from models.model_loader import ModelLoader
from app.config import TOKEN_LIMITS


class Translator:
    """
    Uses the underlying LLM to translate strings from one language to another.
    """

    def __init__(self):
        """Initializes the translator service."""
        self.model = ModelLoader()

    def translate(self, text, target_language):
        """
        Translates text to a specified language.

        Args:
            text (str): The input text to be translated.
            target_language (str): The language to translate the text into (e.g., 'French').

        Returns:
            str: The translated text.
        """

        prompt = f"""[INST] Translate the following text to {target_language}. Output only the translated text, nothing else. Do not add explanations or examples.

        Text: {text} [/INST]"""

        return self.model.generate(prompt, TOKEN_LIMITS["translation"])