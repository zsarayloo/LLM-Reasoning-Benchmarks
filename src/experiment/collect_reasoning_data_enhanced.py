import os
import sys
from typing import List, Tuple, Dict, Any
import random

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
# Local hook utilities
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


def build_cot_prompt(question: str) -> str:
    """
    Enhanced CoT-style prompt for better reasoning.
    """
    return (
        "You are a careful mathematician. Think step by step and solve the problem.\n"
        "Show your work clearly and check your calculations.\n"
        "At the end, provide your final answer.\n\n"
        f"Question: {question}\n\n"
        "Solution:"
    )


def get_hidden_and_response(prompt: str) -> Tuple[torch.Tensor, str, str]:
    """
    Get hidden state and full response for evaluation.
    Returns: (hidden_state, raw_response, extracted_answer)
    """
    with HiddenCapture(model, LAYER) as cap, torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=300,  # Increased for better reasoning
            do_sample=False,              # greedy decoding
            pad_token_id=tokenizer.eos_token_id,
        )
    
    if cap.hidden is None:
        raise RuntimeError("HiddenCapture did not record any hidden states.")
    
    hidden = last_token(cap.hidden)
    raw_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    
    # Extract answer from response
    answer_line = raw_text.splitlines()[-1]
    if "Answer:" in answer_line:
        pred_str = answer_line.split("Answer:", 1)[1].strip()
    elif "=" in answer_line and answer_line.strip().split()[-1].replace(".", "").replace("-", "").isdigit():
        pred_str = answer_line.strip().split()[-1]
    else:
        pred_str = answer_line.strip()
    
    return hidden, raw_text, pred_str


def is_confident_example(raw_response: str, is_correct: bool, confidence_threshold: float = 0.8) -> bool:
    """
    Determine if this is a confident example based on response characteristics.
    For now, use simple heuristics. Can be enhanced with more sophisticated methods.
    """
    response_lower = raw_response.lower()
    
    # For correct examples: be less strict since they're rare
    if is_correct:
        # Accept any response that shows some reasoning attempt
        has_reasoning = any(word in response_lower for word in ["step", "first", "then", "therefore", "thus", "because", "since", "so"])
        has_math = any(char in raw_response for char in ["=", "+", "-", "*", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
        return has_reasoning or has_math or len(raw_response.split()) > 10  # Accept if has reasoning, math, or is detailed
    
    # For incorrect examples: be more selective
    else:
        # Look for clear calculation attempts that are wrong
        has_calculation = any(char in raw_response for char in ["=", "+", "-", "*", "/"])
        return has_calculation and len(raw_response.split()) > 5  # Must have calculations and some detail


def main() -> None:
    # Configuration
    train_split = 0.7  # 70% for training reasoning vector
    confidence_filter = True  # Use only confident examples
    
    print(f"[collect_reasoning_data_enhanced] Loading examples from LiveMathBench")
    
    # 1) Load examples and shuffle for randomness
    all_examples_df = load_livemathbench(split="all")
    print(f"[collect_reasoning_data_enhanced] Found {len(all_examples_df)} total examples")
    
    # Convert DataFrame to list of dictionaries and shuffle
    all_examples = all_examples_df.to_dict('records')
    random.seed(42)  # For reproducibility
    random.shuffle(all_examples)
    
    # Use all available examples
    n_samples = len(all_examples)
    examples = all_examples
    
    # 2) Split into train/test
    n_train = int(n_samples * train_split)
    train_examples = examples[:n_train]
    test_examples = examples[n_train:]
    
    print(f"[collect_reasoning_data_enhanced] Train: {len(train_examples)}, Test: {len(test_examples)}")
    
    # 3) Process training examples
    H_pos_train: List[torch.Tensor] = []
    H_neg_train: List[torch.Tensor] = []
    train_metadata: List[Dict[str, Any]] = []
    
    print("[collect_reasoning_data_enhanced] Processing training examples...")
    for i, ex in enumerate(train_examples, start=1):
        q = ex["question"]
        gold = ex["answer"]
        
        prompt = build_cot_prompt(q)
        
        try:
            # Get hidden state and response
            h, raw_response, pred_str = get_hidden_and_response(prompt)
            
            # Evaluate correctness
            pred_norm = normalize_short_answer_str(pred_str)
            gold_norm = normalize_short_answer_str(gold)
            result = verify_math_answer(pred_norm, gold_norm)
            is_correct = result["correct"]
            
            # Apply confidence filter if enabled
            if confidence_filter:
                if not is_confident_example(raw_response, is_correct):
                    continue
            
            # Store data
            metadata = {
                "idx": i,
                "question": q,
                "gold_answer": gold,
                "predicted_answer": pred_str,
                "raw_response": raw_response,
                "is_correct": is_correct,
                "split": ex.get("split", "unknown")
            }
            train_metadata.append(metadata)
            
            (H_pos_train if is_correct else H_neg_train).append(h.cpu())
            
            if i % 50 == 0 or i == len(train_examples):
                print(
                    f"[collect_reasoning_data_enhanced] Train {i}/{len(train_examples)} "
                    f"(pos={len(H_pos_train)}, neg={len(H_neg_train)})"
                )
                
        except Exception as e:
            print(f"[collect_reasoning_data_enhanced] Error processing example {i}: {e}")
            continue
    
    # 4) Process test examples (for held-out evaluation)
    test_metadata: List[Dict[str, Any]] = []
    
    print("[collect_reasoning_data_enhanced] Processing test examples...")
    for i, ex in enumerate(test_examples, start=1):
        q = ex["question"]
        gold = ex["answer"]
        
        metadata = {
            "idx": i + len(train_examples),
            "question": q,
            "gold_answer": gold,
            "split": ex.get("split", "unknown")
        }
        test_metadata.append(metadata)
        
        if i % 50 == 0 or i == len(test_examples):
            print(f"[collect_reasoning_data_enhanced] Test metadata {i}/{len(test_examples)}")
    
    # 5) Save training data
    if len(H_pos_train) > 0 and len(H_neg_train) > 0:
        H_pos_arr = torch.stack(H_pos_train).numpy()
        H_neg_arr = torch.stack(H_neg_train).numpy()
        
        np.save(os.path.join(PROJECT_ROOT, "H_pos_train.npy"), H_pos_arr)
        np.save(os.path.join(PROJECT_ROOT, "H_neg_train.npy"), H_neg_arr)
        
        print(f"[collect_reasoning_data_enhanced] Saved training data to {PROJECT_ROOT}")
        print(f"  H_pos_train shape: {H_pos_arr.shape}, H_neg_train shape: {H_neg_arr.shape}")
    else:
        print("[collect_reasoning_data_enhanced] WARNING: No positive or negative examples found!")
    
    # 6) Save metadata
    import json
    
    with open(os.path.join(PROJECT_ROOT, "train_metadata.json"), "w") as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(os.path.join(PROJECT_ROOT, "test_metadata.json"), "w") as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"[collect_reasoning_data_enhanced] Saved metadata files")
    print(f"[collect_reasoning_data_enhanced] Summary:")
    print(f"  Total examples processed: {len(train_examples)}")
    print(f"  Confident positive examples: {len(H_pos_train)}")
    print(f"  Confident negative examples: {len(H_neg_train)}")
    print(f"  Test examples for evaluation: {len(test_examples)}")
    
    if confidence_filter:
        pos_rate = len(H_pos_train) / len(train_examples) if len(train_examples) > 0 else 0
        neg_rate = len(H_neg_train) / len(train_examples) if len(train_examples) > 0 else 0
        print(f"  Confidence filtering: pos_rate={pos_rate:.3f}, neg_rate={neg_rate:.3f}")


if __name__ == "__main__":
    main()
