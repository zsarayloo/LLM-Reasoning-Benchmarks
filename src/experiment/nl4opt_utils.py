# src/experiment/nl4opt_utils.py

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ==========================
# Dataset utilities
# ==========================

def load_nl4opt(n_examples: Optional[int] = None,
                random_state: int = 0) -> pd.DataFrame:
    """
    Load NL4OPT_with_optimal_solution.json from Hugging Face (via hf://).

    If n_examples is None -> use FULL dataset (245 problems).
    """
    path = "hf://datasets/CardinalOperations/NL4OPT/NL4OPT_with_optimal_solution.json"
    print(f"[load_nl4opt] Loading dataset from: {path}")
    df = pd.read_json(path, lines=True)

    assert "en_question" in df.columns and "en_answer" in df.columns, \
        "Expected columns 'en_question' and 'en_answer' not found."

    if n_examples is not None:
        df = df.sample(n=n_examples, random_state=random_state).reset_index(drop=True)
        print(f"[load_nl4opt] Loaded {len(df)} examples (sample).")
    else:
        df = df.reset_index(drop=True)
        print(f"[load_nl4opt] Loaded FULL dataset: {len(df)} examples.")

    return df


# ==========================
# Parsing + verification
# ==========================

def extract_last_number(text: str) -> Optional[float]:
    """
    Extract the last numeric value from text.
    Used for parsing the model's final answer.
    """
    if not text:
        return None
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def verify_answer_numeric(pred: Optional[float],
                          gt: float,
                          tol: float = 1e-3) -> Dict[str, Any]:
    """
    Simple numeric verifier: compares a predicted number to the ground-truth
    objective value (for evaluation only; NOT leaked back into the model).
    """
    if pred is None:
        return {"correct": False, "abs_error": None, "rel_error": None}

    abs_err = abs(pred - gt)
    rel_err = abs_err / (abs(gt) + 1e-9)
    return {
        "correct": abs_err <= tol,
        "abs_error": abs_err,
        "rel_error": rel_err,
    }


# ==========================
# GPT-5.1 caller
# ==========================

class GPT5Caller:
    """
    Small wrapper around OpenAI Responses API for GPT-5.1.
    """

    def __init__(self, model_name: str = "gpt-5.1"):
        load_dotenv()
        self.client = OpenAI()
        self.model_name = model_name

    def call(self,
             prompt: str,
             temperature: float = 0.0) -> str:
        """
        Low-level wrapper to call gpt-5.1 with a plain text prompt.
        """
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=temperature,
        )
        # Assumes simple text output (no tools, no images)
        return response.output[0].content[0].text


# ==========================
# PoT-related utilities
# ==========================

def build_pot_prompt(question: str) -> str:
    """
    Program-of-Thoughts: ask GPT-5.1 to write a small Python function solve()
    that returns the objective value.
    """
    return (
        "You are an expert in mathematical optimization and Python programming.\n"
        "You will solve the following linear programming word problem by writing a Python function.\n\n"
        "Requirements:\n"
        "1. Write a Python function with signature `def solve():` that returns the optimal objective value as a float.\n"
        "2. Do NOT use any external libraries (no pulp, no cvxpy, etc.).\n"
        "3. You may use basic Python (loops, arithmetic, if, etc.) to compute the answer.\n"
        "4. Do NOT print anything. Just compute and return the value.\n"
        "5. Only output the Python code, enclosed in a single code block starting with ```python and ending with ```.\n\n"
        "Problem:\n"
        f"{question}\n"
    )


def extract_code_block(text: str) -> Optional[str]:
    """
    Extract the first ```python ... ``` code block. If not found, try bare ``` ... ``` block.
    """
    if not text:
        return None
    m = re.search(r"```python(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def execute_pot_code(code: str) -> Optional[float]:
    """
    Execute the model-generated Python code in a minimal namespace and call solve().

    NOTE: This executes arbitrary code from the model.
    In a real system, sandbox this. Here we trust it for research purposes.
    """
    local_ns: Dict[str, Any] = {}
    try:
        exec(code, {}, local_ns)  # no globals, only locals
        if "solve" not in local_ns:
            return None
        result = local_ns["solve"]()
        return float(result)
    except Exception as e:
        print("[execute_pot_code] Error executing code:", repr(e))
        return None


# ==========================
# Simple summary helper
# ==========================

def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize accuracy and errors for a single-strategy run.
    Expects columns: correct, abs_error, rel_error, latency_sec.
    """
    n = len(df)
    n_ok = int(df["correct"].sum())
    acc = df["correct"].mean()
    mean_abs = df["abs_error"].mean()
    mean_rel = df["rel_error"].mean()
    mean_lat = df["latency_sec"].mean()

    summary = pd.DataFrame([{
        "n": n,
        "n_ok": n_ok,
        "acc": acc,
        "mean_abs_err": mean_abs,
        "mean_rel_err": mean_rel,
        "mean_latency": mean_lat,
    }])

    return summary
