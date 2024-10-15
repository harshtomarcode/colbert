import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    system_memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent()

    logging.info(f"Process Memory: Used = {mem_info.rss / 1024 / 1024:.2f} MB")
    logging.info(f"System Memory: Total = {system_memory.total / 1024 / 1024:.2f} MB, "
                 f"Available = {system_memory.available / 1024 / 1024:.2f} MB, "
                 f"Used = {system_memory.used / 1024 / 1024:.2f} MB, "
                 f"Percent = {system_memory.percent:.1f}%")
    logging.info(f"CPU Usage: {cpu_percent:.1f}%")

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
        logging.info(f"Model saved to {model_dir}")
    else:
        logging.info(f"Loading model from {model_dir}")
