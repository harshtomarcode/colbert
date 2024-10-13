import torch
from transformers import pipeline
from utils import load_model
from dotenv import load_dotenv

model_dir = "./model/"

load_dotenv()
load_model(model_dir)

pipe = pipeline(
    "text-generation",
    model=model_dir,
    tokenizer=model_dir,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cpu",
)

# messages = [
#     {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
# ]

def get_response(messages):
    outputs = pipe(messages, max_new_tokens=256)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    return assistant_response