import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llama3_3b(device="cuda"):
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
    return tokenizer, model
