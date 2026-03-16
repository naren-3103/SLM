import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.config import MODEL_PATH, GENERATION_CONFIG


class ModelLoader:

    def __init__(self):

        # Detect device automatically
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )

        self.model.to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def generate(self, prompt, max_tokens):

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=GENERATION_CONFIG["temperature"],
                top_p=GENERATION_CONFIG["top_p"],
                top_k=GENERATION_CONFIG["top_k"],
                do_sample=GENERATION_CONFIG["do_sample"],
                pad_token_id=self.tokenizer.eos_token_id
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        generated = text[len(prompt):].strip()

        return generated.split("\n")[0]