import os
import sys
from typing import List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------
# Add project root + src/ to sys.path so we can import `model` and `utils`
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

for p in (PROJECT_ROOT, SRC_ROOT):
    if p not in sys.path:
        sys.path.append(p)

from model.lama3_3b_loader import load_llama3_3b  # uses ollama-backed llama3.2:3b
from utils.capture_hidden import HiddenCapture, last_token  # hook utils

# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------
tokenizer, model = load_llama3_3b()

LAYER = 12  # mid-layer to probe
H_pos: List[torch.Tensor] = []
H_neg: List[torch.Tensor] = []


def build_cot_prompt(question: str) -> str:
    """
    Simple CoT-style prompt for math/logic questions.
    You can later swap this for a more advanced prompt.
    """
    return (
        "You are a careful mathematician. Think step by step and solve the problem.\n\n"
        f"Question: {question}\n\n"
        "Reasoning:"
    )


def get_hidden(prompt: str) -> torch.Tensor:
    """
    Run the model with a HiddenCapture hook on LAYER and return
    the last-token hidden state at that layer.
    """
    with HiddenCapture(model, LAYER) as cap:
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        _ = model.generate(**ids, max_new_tokens=100, temperature=0.0)
    return last_token(cap.hidden)


# ---------------------------------------------------------------------
# TEMP: toy dataset just to exercise the pipeline
# Later you can replace this with (question, is_correct) pairs from
# LiveMathBench or NL4OPT.
# ---------------------------------------------------------------------
toy_dataset: List[Tuple[str, bool]] = [
    ("If x + y = 10 and xy = 21, find x^2 + y^2.", True),
    ("Is 2 + 2 = 5?", False),
    ("Solve: 3x + 4 = 10. What is x?", True),
    ("Is 1/0 finite?", False),
]


def main() -> None:
    global H_pos, H_neg

    for question, is_correct in toy_dataset:
        prompt = build_cot_prompt(question)
        h = get_hidden(prompt)
        if is_correct:
            H_pos.append(h.cpu())
        else:
            H_neg.append(h.cpu())

    # Stack and save in the project root (same place you'll run the script)
    H_pos_arr = torch.stack(H_pos).numpy()
    H_neg_arr = torch.stack(H_neg).numpy()

    np.save(os.path.join(PROJECT_ROOT, "H_pos.npy"), H_pos_arr)
    np.save(os.path.join(PROJECT_ROOT, "H_neg.npy"), H_neg_arr)

    print(f"Saved H_pos.npy and H_neg.npy to {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
