# model/lama3_3b_loader.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

def load_llama3_3b(device="cpu"):
    print(f"[lama3_3b_loader] Loading {MODEL_NAME} on device={device}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map={"": device},   # <-- critical: force CPU
        torch_dtype=torch.float32, # <-- CPU-friendly, no half precision!
    )

    model.eval()
    return tokenizer, model
