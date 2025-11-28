import os
import sys
from typing import List, Tuple, Dict, Any
import random

import numpy as np
import torch

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
    Enhanced CoT-style prompt for harder math problems.
    """
    return (
        "You are a skilled mathematician. Solve this problem step by step.\n"
        "Show your reasoning clearly and double-check your work.\n"
        "At the end, provide your final numerical answer.\n\n"
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
            max_new_tokens=400,  # More tokens for complex reasoning
            do_sample=False,              # greedy decoding
            pad_token_id=tokenizer.eos_token_id,
        )
    
    if cap.hidden is None:
        raise RuntimeError("HiddenCapture did not record any hidden states.")
    
    hidden = last_token(cap.hidden)
    raw_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    
    # Extract answer from response - look for numbers
    response = raw_text[len(prompt):].strip()
    
    # Try to find the final number in the response
    import re
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        pred_str = numbers[-1]  # Take the last number as the answer
    else:
        pred_str = response.split()[-1] if response.split() else ""
    
    return hidden, response, pred_str


def create_harder_math_problems() -> List[Dict[str, Any]]:
    """
    Create harder math problems that will challenge the 3B model more.
    """
    problems = []
    
    # Multi-step arithmetic (harder)
    for i in range(15):
        a = random.randint(15, 50)
        b = random.randint(10, 30)
        c = random.randint(5, 20)
        question = f"What is ({a} + {b}) × {c}?"
        answer = str((a + b) * c)
        problems.append({"question": question, "answer": answer, "type": "multi_step"})
    
    # Division with remainders
    for i in range(15):
        dividend = random.randint(50, 200)
        divisor = random.randint(7, 15)
        question = f"What is {dividend} ÷ {divisor}? Give the whole number result."
        answer = str(dividend // divisor)
        problems.append({"question": question, "answer": answer, "type": "division"})
    
    # Percentage problems
    percentage_problems = [
        {"question": "What is 25% of 80?", "answer": "20", "type": "percentage"},
        {"question": "What is 15% of 120?", "answer": "18", "type": "percentage"},
        {"question": "What is 30% of 60?", "answer": "18", "type": "percentage"},
        {"question": "What is 40% of 75?", "answer": "30", "type": "percentage"},
        {"question": "What is 20% of 150?", "answer": "30", "type": "percentage"},
        {"question": "What is 35% of 40?", "answer": "14", "type": "percentage"},
        {"question": "What is 50% of 90?", "answer": "45", "type": "percentage"},
        {"question": "What is 10% of 250?", "answer": "25", "type": "percentage"},
    ]
    problems.extend(percentage_problems)
    
    # Complex word problems
    complex_word_problems = [
        {"question": "A store has 144 apples. They sell 3/4 of them. How many apples are left?", "answer": "36", "type": "fraction_word"},
        {"question": "If a car travels 60 miles per hour for 2.5 hours, how many miles does it travel?", "answer": "150", "type": "rate_word"},
        {"question": "A rectangle has length 12 and width 8. What is its area?", "answer": "96", "type": "geometry"},
        {"question": "If you buy 3 items for $15 each and pay with a $50 bill, how much change do you get?", "answer": "5", "type": "money_word"},
        {"question": "A box contains 48 chocolates arranged in 6 equal rows. How many chocolates are in each row?", "answer": "8", "type": "division_word"},
        {"question": "If a train travels 240 miles in 4 hours, what is its average speed in miles per hour?", "answer": "60", "type": "rate_word"},
        {"question": "A pizza is cut into 8 equal slices. If you eat 3 slices, what fraction of the pizza is left?", "answer": "5", "type": "fraction_word"},  # 5/8, but we'll accept 5
        {"question": "If you save $12 per week for 6 weeks, how much money do you save in total?", "answer": "72", "type": "multiplication_word"},
        {"question": "A classroom has 5 rows of desks with 6 desks in each row. If 4 desks are empty, how many desks are occupied?", "answer": "26", "type": "multi_step_word"},
        {"question": "If a book has 240 pages and you read 15 pages per day, how many days will it take to finish?", "answer": "16", "type": "division_word"},
    ]
    problems.extend(complex_word_problems)
    
    # Algebraic thinking (simple)
    algebra_problems = [
        {"question": "If x + 7 = 15, what is x?", "answer": "8", "type": "algebra"},
        {"question": "If 3y = 21, what is y?", "answer": "7", "type": "algebra"},
        {"question": "If 2a + 4 = 14, what is a?", "answer": "5", "type": "algebra"},
        {"question": "If x - 9 = 12, what is x?", "answer": "21", "type": "algebra"},
        {"question": "If 4b = 32, what is b?", "answer": "8", "type": "algebra"},
    ]
    problems.extend(algebra_problems)
    
    return problems


def simple_verify_answer(predicted: str, correct: str) -> bool:
    """
    Simple answer verification for math problems.
    """
    try:
        pred_num = float(predicted.strip())
        correct_num = float(correct.strip())
        return abs(pred_num - correct_num) < 0.01
    except:
        return predicted.strip().lower() == correct.strip().lower()


def main() -> None:
    print(f"[collect_reasoning_data_hard] Creating harder math problems for 3B model")
    
    # 1) Create harder problems
    all_problems = create_harder_math_problems()
    random.seed(42)  # For reproducibility
    random.shuffle(all_problems)
    
    print(f"[collect_reasoning_data_hard] Created {len(all_problems)} harder math problems")
    
    # 2) Split into train/test
    train_split = 0.7
    n_train = int(len(all_problems) * train_split)
    train_examples = all_problems[:n_train]
    test_examples = all_problems[n_train:]
    
    print(f"[collect_reasoning_data_hard] Train: {len(train_examples)}, Test: {len(test_examples)}")
    
    # 3) Process training examples
    H_pos_train: List[torch.Tensor] = []
    H_neg_train: List[torch.Tensor] = []
    train_metadata: List[Dict[str, Any]] = []
    
    print("[collect_reasoning_data_hard] Processing training examples...")
    for i, ex in enumerate(train_examples, start=1):
        question = ex["question"]
        correct_answer = ex["answer"]
        
        prompt = build_cot_prompt(question)
        
        try:
            # Get hidden state and response
            h, raw_response, pred_str = get_hidden_and_response(prompt)
            
            # Evaluate correctness
            is_correct = simple_verify_answer(pred_str, correct_answer)
            
            # Store data (no confidence filtering for harder problems)
            metadata = {
                "idx": i,
                "question": question,
                "gold_answer": correct_answer,
                "predicted_answer": pred_str,
                "raw_response": raw_response,
                "is_correct": is_correct,
                "type": ex["type"]
            }
            train_metadata.append(metadata)
            
            (H_pos_train if is_correct else H_neg_train).append(h.cpu())
            
            if i % 20 == 0 or i == len(train_examples):
                print(
                    f"[collect_reasoning_data_hard] Train {i}/{len(train_examples)} "
                    f"(pos={len(H_pos_train)}, neg={len(H_neg_train)})"
                )
                
        except Exception as e:
            print(f"[collect_reasoning_data_hard] Error processing example {i}: {e}")
            continue
    
    # 4) Process test examples (for held-out evaluation)
    test_metadata: List[Dict[str, Any]] = []
    
    print("[collect_reasoning_data_hard] Processing test examples...")
    for i, ex in enumerate(test_examples, start=1):
        metadata = {
            "idx": i + len(train_examples),
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "type": ex["type"]
        }
        test_metadata.append(metadata)
    
    print(f"[collect_reasoning_data_hard] Test metadata: {len(test_metadata)} examples")
    
    # 5) Save training data
    if len(H_pos_train) > 0 and len(H_neg_train) > 0:
        H_pos_arr = torch.stack(H_pos_train).numpy()
        H_neg_arr = torch.stack(H_neg_train).numpy()
        
        np.save(os.path.join(PROJECT_ROOT, "H_pos_train.npy"), H_pos_arr)
        np.save(os.path.join(PROJECT_ROOT, "H_neg_train.npy"), H_neg_arr)
        
        print(f"[collect_reasoning_data_hard] Saved training data to {PROJECT_ROOT}")
        print(f"  H_pos_train shape: {H_pos_arr.shape}, H_neg_train shape: {H_neg_arr.shape}")
    else:
        print("[collect_reasoning_data_hard] WARNING: No positive or negative examples found!")
        # Save whatever we have
        if len(H_pos_train) > 0:
            H_pos_arr = torch.stack(H_pos_train).numpy()
            np.save(os.path.join(PROJECT_ROOT, "H_pos_train.npy"), H_pos_arr)
            print(f"  H_pos_train shape: {H_pos_arr.shape}")
        if len(H_neg_train) > 0:
            H_neg_arr = torch.stack(H_neg_train).numpy()
            np.save(os.path.join(PROJECT_ROOT, "H_neg_train.npy"), H_neg_arr)
            print(f"  H_neg_train shape: {H_neg_arr.shape}")
    
    # 6) Save metadata
    import json
    
    with open(os.path.join(PROJECT_ROOT, "train_metadata.json"), "w") as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(os.path.join(PROJECT_ROOT, "test_metadata.json"), "w") as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"[collect_reasoning_data_hard] Saved metadata files")
    print(f"[collect_reasoning_data_hard] Summary:")
    print(f"  Total examples processed: {len(train_examples)}")
    print(f"  Correct examples: {len(H_pos_train)}")
    print(f"  Incorrect examples: {len(H_neg_train)}")
    print(f"  Test examples for evaluation: {len(test_examples)}")
    
    if len(train_examples) > 0:
        accuracy = len(H_pos_train) / len(train_examples)
        print(f"  Model accuracy on harder problems: {accuracy:.3f}")


if __name__ == "__main__":
    main()
