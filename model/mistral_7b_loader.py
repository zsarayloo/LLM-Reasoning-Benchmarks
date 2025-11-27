# model/mistral_7b_loader.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_mistral_7b(device: str | None = None):
    """
    Load Mistral-7B-Instruct-v0.1 locally via transformers.

    Uses:
      - device='cuda' if available (GPU node), else 'cpu'
      - fp16 on CUDA, fp32 on CPU
    """
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[mistral_7b_loader] Loading {model_name} on device={device}")

    # HF_TOKEN is picked up automatically by transformers/hf_hub, no need to pass explicitly.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    kwargs = {}
    if device == "cuda":
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    return tokenizer, model
