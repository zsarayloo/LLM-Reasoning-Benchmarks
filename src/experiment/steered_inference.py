import numpy as np
import torch

from models.llama3_3b_loader import load_llama3_3b
from utils.steer import SteeringHook

tokenizer, model = load_llama3_3b()
c = np.load("reasoning_vector.npy")

LAYER = 12
ALPHA = 1.5

def answer(query):
    prompt = f"Think step by step and solve:\n\n{query}\n\nAnswer:"
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    with SteeringHook(model, LAYER, c, ALPHA):
        out = model.generate(**ids, max_new_tokens=200, temperature=0)

    return tokenizer.decode(out[0], skip_special_tokens=True)

print(answer("If x+y=10 and xy=21, find x^2 + y^2."))
