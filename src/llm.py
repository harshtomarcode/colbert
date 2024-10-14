import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from utils import load_model
from dotenv import load_dotenv
import yaml

model_dir = "./model/"

load_dotenv()
load_model(model_dir)

model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

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

def load_prompt(file_path):
    with open(file_path, 'r') as file:
        prompt_data = yaml.safe_load(file)
    return prompt_data['system_prompt']

def generate_response(prompt_file, context, conversation):
    system_prompt = load_prompt(prompt_file)
    
    # Format the conversation history
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
    
    full_prompt = system_prompt.format(context_snippets=context, user_query=conversation_history)
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    assistant_response = response.split("Response:")[-1].strip()
    
    # Update the conversation state with the assistant's response
    conversation = manage_conversation_state(conversation, "assistant", assistant_response)
    
    return assistant_response, conversation

def manage_conversation_state(conversation, role, content):
    conversation.append({"role": role, "content": content})
    return conversation

# Example usage:
# response = generate_response("src/prompts/response.yml", context, query)
