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
    Simple CoT-style prompt for easy math problems.
    """
    return (
        "You are a helpful assistant. Solve this simple math problem step by step.\n\n"
        f"Question: {question}\n\n"
        "Let me solve this step by step:\n"
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
            max_new_tokens=150,  # Shorter for simple problems
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


def create_simple_math_problems() -> List[Dict[str, Any]]:
    """
    Create simple math problems that a 3B model can solve.
    """
    problems = []
    
    # Simple addition problems
    for i in range(20):
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        question = f"What is {a} + {b}?"
        answer = str(a + b)
        problems.append({"question": question, "answer": answer, "type": "addition"})
    
    # Simple subtraction problems
    for i in range(20):
        a = random.randint(10, 30)
        b = random.randint(1, 9)
        question = f"What is {a} - {b}?"
        answer = str(a - b)
        problems.append({"question": question, "answer": answer, "type": "subtraction"})
    
    # Simple multiplication problems
    for i in range(20):
        a = random.randint(2, 10)
        b = random.randint(2, 10)
        question = f"What is {a} × {b}?"
        answer = str(a * b)
        problems.append({"question": question, "answer": answer, "type": "multiplication"})
    
    # Simple word problems
    word_problems = [
        {"question": "If I have 5 apples and buy 3 more, how many apples do I have?", "answer": "8", "type": "word"},
        {"question": "There are 12 birds on a tree. 4 fly away. How many are left?", "answer": "8", "type": "word"},
        {"question": "A box has 6 red balls and 4 blue balls. How many balls in total?", "answer": "10", "type": "word"},
        {"question": "If each pizza has 8 slices and I have 2 pizzas, how many slices total?", "answer": "16", "type": "word"},
        {"question": "I start with 20 dollars and spend 7 dollars. How much money do I have left?", "answer": "13", "type": "word"},
        {"question": "A classroom has 4 rows of desks with 5 desks in each row. How many desks total?", "answer": "20", "type": "word"},
        {"question": "If I read 3 pages per day for 5 days, how many pages did I read?", "answer": "15", "type": "word"},
        {"question": "There are 15 students and 3 leave. How many students remain?", "answer": "12", "type": "word"},
        {"question": "I have 2 bags with 7 marbles each. How many marbles do I have?", "answer": "14", "type": "word"},
        {"question": "A parking lot has 25 cars. 8 cars leave. How many cars are still there?", "answer": "17", "type": "word"},
    ]
    problems.extend(word_problems)
    
    return problems


def simple_verify_answer(predicted: str, correct: str) -> bool:
    """
    Simple answer verification for basic math problems.
    """
    try:
        pred_num = float(predicted.strip())
        correct_num = float(correct.strip())
        return abs(pred_num - correct_num) < 0.01
    except:
        return predicted.strip().lower() == correct.strip().lower()


def main() -> None:
    print(f"[collect_reasoning_data_simple] Creating simple math problems for 3B model")
    
    # 1) Create simple problems
    all_problems = create_simple_math_problems()
    random.seed(42)  # For reproducibility
    random.shuffle(all_problems)
    
    print(f"[collect_reasoning_data_simple] Created {len(all_problems)} simple math problems")
    
    # 2) Split into train/test
    train_split = 0.7
    n_train = int(len(all_problems) * train_split)
    train_examples = all_problems[:n_train]
    test_examples = all_problems[n_train:]
    
    print(f"[collect_reasoning_data_simple] Train: {len(train_examples)}, Test: {len(test_examples)}")
    
    # 3) Process training examples
    H_pos_train: List[torch.Tensor] = []
    H_neg_train: List[torch.Tensor] = []
    train_metadata: List[Dict[str, Any]] = []
    
    print("[collect_reasoning_data_simple] Processing training examples...")
    for i, ex in enumerate(train_examples, start=1):
        question = ex["question"]
        correct_answer = ex["answer"]
        
        prompt = build_cot_prompt(question)
        
        try:
            # Get hidden state and response
            h, raw_response, pred_str = get_hidden_and_response(prompt)
            
            # Evaluate correctness
            is_correct = simple_verify_answer(pred_str, correct_answer)
            
            # Store data (no confidence filtering for simple problems)
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
                    f"[collect_reasoning_data_simple] Train {i}/{len(train_examples)} "
                    f"(pos={len(H_pos_train)}, neg={len(H_neg_train)})"
                )
                
        except Exception as e:
            print(f"[collect_reasoning_data_simple] Error processing example {i}: {e}")
            continue
    
    # 4) Process test examples (for held-out evaluation)
    test_metadata: List[Dict[str, Any]] = []
    
    print("[collect_reasoning_data_simple] Processing test examples...")
    for i, ex in enumerate(test_examples, start=1):
        metadata = {
            "idx": i + len(train_examples),
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "type": ex["type"]
        }
        test_metadata.append(metadata)
    
    print(f"[collect_reasoning_data_simple] Test metadata: {len(test_metadata)} examples")
    
    # 5) Save training data
    if len(H_pos_train) > 0 and len(H_neg_train) > 0:
        H_pos_arr = torch.stack(H_pos_train).numpy()
        H_neg_arr = torch.stack(H_neg_train).numpy()
        
        np.save(os.path.join(PROJECT_ROOT, "H_pos_train.npy"), H_pos_arr)
        np.save(os.path.join(PROJECT_ROOT, "H_neg_train.npy"), H_neg_arr)
        
        print(f"[collect_reasoning_data_simple] Saved training data to {PROJECT_ROOT}")
        print(f"  H_pos_train shape: {H_pos_arr.shape}, H_neg_train shape: {H_neg_arr.shape}")
    else:
        print("[collect_reasoning_data_simple] WARNING: No positive or negative examples found!")
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
    
    print(f"[collect_reasoning_data_simple] Saved metadata files")
    print(f"[collect_reasoning_data_simple] Summary:")
    print(f"  Total examples processed: {len(train_examples)}")
    print(f"  Correct examples: {len(H_pos_train)}")
    print(f"  Incorrect examples: {len(H_neg_train)}")
    print(f"  Test examples for evaluation: {len(test_examples)}")
    
    if len(train_examples) > 0:
        accuracy = len(H_pos_train) / len(train_examples)
        print(f"  Model accuracy on simple problems: {accuracy:.3f}")


if __name__ == "__main__":
    main()
