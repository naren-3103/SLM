"""
Service for summarizing raw text inputs.
"""
from models.model_loader import ModelLoader
from app.config import TOKEN_LIMITS


class TextSummarizer:
    """
    Wraps the LLM for concise text summarization tasks.
    """

    def __init__(self):
        """Initializes the TextSummarizer with the model instance."""
        self.model = ModelLoader()

    def summarize_text(self, text):
        """
        Generates a summary of the provided text.

        Args:
            text (str): The text to summarize.

        Returns:
            str: The model-generated summary.
        """
        prompt = f"""[INST] Summarize the following text in a few concise sentences. Output only the summary, nothing else.

    Text: {text} [/INST]"""

        return self.model.generate(prompt, TOKEN_LIMITS["summarization"])