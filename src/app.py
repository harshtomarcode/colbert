import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()
# Set the directory where the model will be saved
model_dir = "./model/"

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

# Load the model and tokenizer from the saved directory
pipe = pipeline(
    "text-generation",
    model=model_dir,
    tokenizer=model_dir,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cpu",
)

messages = [
    {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
]

outputs = pipe(messages, max_new_tokens=256)
assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
print(assistant_response)
