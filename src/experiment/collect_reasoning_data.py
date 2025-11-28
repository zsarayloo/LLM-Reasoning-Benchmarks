import os
import sys
from typing import List, Tuple

import numpy as np
import torch

from experiment.data_loader_livemathbench import load_livemathbench
from experiment.analyze_livemathbench_cot_vs_sc import (
    normalize_answer_str as normalize_short_answer_str,
    verify_math_answer,
)

# ---------------------------------------------------------------------
# Make sure we can import from project root and src/
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

for p in (PROJECT_ROOT, SRC_ROOT):
    if p not in sys.path:
        sys.path.append(p)

from model.lama3_3b_loader import load_llama3_3b  # Llama-3.2-3B loader


# ---------------------------------------------------------------------
# Local hook utilities (so we don't depend on utils.capture_hidden)
# ---------------------------------------------------------------------
class HiddenCapture:
    """
    Context manager that registers a forward hook on a given decoder layer
    and stores its output hidden states in self.hidden.
    """

    def __init__(self, model: torch.nn.Module, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.handle = None
        self.hidden = None

    def __enter__(self):
        # For Llama, decoder layers live under model.model.layers
        layer = self.model.model.layers[self.layer_idx]

        def hook(module, inputs, output):
            # For LlamaDecoderLayer, output is usually a Tensor of shape [B, T, H]
            # (or a tuple whose first element is the tensor). Handle both.
            if isinstance(output, torch.Tensor):
                self.hidden = output
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                self.hidden = output[0]
            else:
                self.hidden = None

        self.handle = layer.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None


def last_token(h: torch.Tensor) -> torch.Tensor:
    """
    Take [B, T, H] hidden states and return [H] for the last token of batch 0.
    """
    # h shape: (batch_size, seq_len, hidden_size)
    return h[0, -1, :]


# ---------------------------------------------------------------------
# Load model once
# ---------------------------------------------------------------------
tokenizer, model = load_llama3_3b()
LAYER = 12  # which decoder layer to probe (0-based index)

H_pos: List[torch.Tensor] = []
H_neg: List[torch.Tensor] = []


def build_cot_prompt(question: str) -> str:
    """
    Simple CoT-style prompt.
    """
    return (
        "You are a careful mathematician. Think step by step and solve the problem.\n\n"
        f"Question: {question}\n\n"
        "Reasoning:"
    )

def get_hidden(prompt: str) -> torch.Tensor:
    with HiddenCapture(model, LAYER) as cap, torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        _ = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,              # greedy decoding
            pad_token_id=tokenizer.eos_token_id,  # optional, kills the warning
        )
    if cap.hidden is None:
        raise RuntimeError("HiddenCapture did not record any hidden states.")
    return last_token(cap.hidden)



# ---------------------------------------------------------------------
# TEMP: tiny toy dataset to exercise the pipeline
# ---------------------------------------------------------------------
#toy_dataset: List[Tuple[str, bool]] = [
#    ("If x + y = 10 and xy = 21, find x^2 + y^2.", True),
#    ("Is 2 + 2 = 5?", False),
#    ("Solve: 3x + 4 = 10. What is x?", True),
#    ("Is 1/0 finite?", False),
#]
def main() -> None:
    global H_pos, H_neg

    # 1) Load a larger subset of LiveMathBench for robust reasoning vector
    all_examples = load_livemathbench(split="all")
    n_samples = 200  # Increased from 50 for better signal
    examples = all_examples[:n_samples]

    print(f"[collect_reasoning_data] Using {len(examples)} examples from LiveMathBench")

    for i, ex in enumerate(examples, start=1):
        q = ex["question"]
        gold = ex["gold_answer"]

        prompt = build_cot_prompt(q)

        # 2) Run model with hook to capture hidden state
        h = get_hidden(prompt)  # [hidden_dim]

        # 3) Generate answer text again (without hook) to evaluate correctness
        #    (Cheap extra forward pass; keeps capture code clean.)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # 4) Extract final answer from raw_text using the same pattern as your CoT script
        #    Here we expect the last line to contain "Answer: ..."
        answer_line = raw_text.splitlines()[-1]
        if "Answer:" in answer_line:
            pred_str = answer_line.split("Answer:", 1)[1].strip()
        else:
            pred_str = answer_line.strip()

        pred_norm = normalize_short_answer_str(pred_str)
        gold_norm = normalize_short_answer_str(gold)
        ok, _ = verify_math_answer(pred_norm, gold_norm)

        (H_pos if ok else H_neg).append(h.cpu())

        if i % 10 == 0 or i == len(examples):
            print(
                f"[collect_reasoning_data] {i}/{len(examples)} "
                f"(pos={len(H_pos)}, neg={len(H_neg)})"
            )

    # 5) Stack and save
    H_pos_arr = torch.stack(H_pos).numpy()
    H_neg_arr = torch.stack(H_neg).numpy()

    np.save(os.path.join(PROJECT_ROOT, "H_pos.npy"), H_pos_arr)
    np.save(os.path.join(PROJECT_ROOT, "H_neg.npy"), H_neg_arr)

    print(f"[collect_reasoning_data] Saved H_pos.npy and H_neg.npy to {PROJECT_ROOT}")
    print(f"  H_pos shape: {H_pos_arr.shape}, H_neg shape: {H_neg_arr.shape}")
