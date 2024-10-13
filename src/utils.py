import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_dir):
    # Check if the model is already saved locally
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

        # Download and save the model and tokenizer if not already saved
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it",
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

        # Save model and tokenizer to the specified directory
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Model saved to {model_dir}")
    else:
        print(f"Loading model from {model_dir}")

