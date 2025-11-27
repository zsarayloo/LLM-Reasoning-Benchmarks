# model/lama3_3b_loader.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

def load_llama3_3b(device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[lama3_3b_loader] Loading {MODEL_NAME} on device={device}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_auth_token=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()
    return tokenizer, model
