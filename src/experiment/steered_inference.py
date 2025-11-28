import os
import sys
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if p not in sys.path:
        sys.path.append(p)

from model.lama3_3b_loader import load_llama3_3b  # Llama-3.2-3B loader


# ---------------------------------------------------------------------
# Load reasoning vector
# ---------------------------------------------------------------------
REASON_VEC_PATH = os.path.join(PROJECT_ROOT, "model", "reasoning_vector.npy")

if not os.path.exists(REASON_VEC_PATH):
    raise FileNotFoundError(f"reasoning_vector.npy not found at {REASON_VEC_PATH}")

v_np = np.load(REASON_VEC_PATH)  # shape (hidden_dim,)
v = torch.from_numpy(v_np).float()
v = v / (v.norm() + 1e-8)        # normalize


# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------
tokenizer, model = load_llama3_3b()
device = model.device
v = v.to(device)

# pick the same mid-layer index used during collection
LAYER_IDX = 12


def build_cot_prompt(question: str) -> str:
    return (
        "You are an expert competition mathematician.\n"
        "Solve the problem step by step with clear reasoning.\n"
        "At the very end, on the LAST line, write exactly:\n"
        "Answer: <final numeric answer>\n"
        "Replace <final numeric answer> with a single number or simple expression.\n"
        "Do NOT leave the placeholder text.\n"
        "Do NOT add anything after the answer on that line.\n\n"
        f"Problem:\n{question}\n\n"
    )


def _make_steering_hook(alpha: float):
    """
    Returns a forward hook that adds alpha * v to the last token
    at layer LAYER_IDX.
    """
    def hook(module, inputs, output):
        # output can be Tensor or tuple; handle both cases
        if isinstance(output, torch.Tensor):
            hidden = output.clone()
            # add to last time step for all batch items
            hidden[:, -1, :] = hidden[:, -1, :] + alpha * v
            return hidden
        elif isinstance(output, (tuple, list)) and len(output) > 0:
            hidden = output[0].clone()
            # add to last time step for all batch items
            hidden[:, -1, :] = hidden[:, -1, :] + alpha * v
            # return tuple with modified first element
            return (hidden,) + output[1:]
        else:
            return output

    return hook


def generate_with_optional_steering(
    question: str,
    alpha: Optional[float] = None,
    max_new_tokens: int = 256,
) -> str:
    """
    Run CoT on `question`, optionally steering with coefficient `alpha`.
    alpha = None or 0.0 -> no steering.
    """
    prompt = build_cot_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    hook_handle = None
    if alpha is not None and alpha != 0.0:
        target_layer = model.model.layers[LAYER_IDX]
        hook_handle = target_layer.register_forward_hook(_make_steering_hook(alpha))

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,     # greedy for stability
        )

    if hook_handle is not None:
        hook_handle.remove()

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # return only the model's continuation, not the whole prompt
    return full_text[len(prompt):].strip()


def demo():
    # Pick a few of the LiveMathBench questions that LLaMA struggles with.
    questions = [
        "What is the value of 9901 * 101 - 99 * 10101?",
        "The number 2024 is written as the sum of not necessarily distinct two-digit numbers. "
        "What is the least number of two-digit numbers needed to write this sum?",
    ]

    alphas = [0.0, 3.0, -3.0]

    for q in questions:
        print("\n" + "=" * 80)
        print("QUESTION:")
        print(q)
        for a in alphas:
            tag = f"alpha={a}" if a != 0.0 else "baseline (no steering)"
            print(f"\n--- {tag} ---")
            resp = generate_with_optional_steering(q, alpha=a)
            print(resp)


if __name__ == "__main__":
    demo()
