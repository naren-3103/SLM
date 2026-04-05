"""
Model Loader module for handling the downloading and initializing of the language model natively without C++.
"""
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from app.config import HF_TOKEN, GENERATION_CONFIG, REPO_ID, MODEL_NAME


class ModelLoader:
    """
    Singleton class to load and provide inference using the Llama model.
    Ensures the model is only loaded into memory once.
    """

    _instance = None  # singleton — model loads only once

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initializes the model by downloading (if necessary) and loading it into memory."""

        print("Downloading GGUF model (~4GB)...")

        model_file = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_NAME,
            token=HF_TOKEN if HF_TOKEN else None,
            cache_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))
        )

        print(f"Model file ready: {model_file}")
        print("Loading into memory...")

        self.model = Llama(
            model_path=model_file,
            n_ctx=4096,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False
        )

        self.tokenizer = type("T", (), {
            "eos_token_id": 2,
            "pad_token_id": 2
        })()

        print("Model loaded successfully!")

    def generate(self, prompt, max_tokens, stop=None):
        """Generic generate — used by all services."""
        
        if stop is None:
            stop = [
                "</s>",
                "[INST]",
                "[/INST]",
                "\nText:",
                "\nTranslation:",
                "\nQuestion:",
                "\nAnswer:",
                "\nSummary:",
                "\n\n",
                "<|user|>",
                "<|system|>"
            ]

        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=GENERATION_CONFIG["temperature"],
            top_p=GENERATION_CONFIG["top_p"],
            top_k=GENERATION_CONFIG["top_k"],
            echo=False,
            stop=stop
        )

        generated_text = output["choices"][0]["text"].strip()
                    
        # Remove common repeating generation prefixes 
        prefixes_to_strip = ["Translation:", "Summary:", "Answer:", "\nTranslation:", "Text:"]
        for p in prefixes_to_strip:
            if generated_text.startswith(p):
                generated_text = generated_text[len(p):].strip()
                    
        return generated_text.strip()