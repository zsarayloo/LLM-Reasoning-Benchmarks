import os
import re
import time
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. Local LiveMathBench loader (all English splits)
# ============================================================

# Mapping from logical split name to local filename
LMB_SPLIT_TO_FILE = {
    "CNMO": "CNMO_en.jsonl",
    "CCEE": "CCEE_en.jsonl",
    "AMC": "AMC_en.jsonl",
    "WLPMC": "WLPMC_en.jsonl",
    "HARD": "hard_en.jsonl",
}


def get_lmb_data_dir() -> str:
    """
    Resolve the LiveMathBench data directory relative to the repo root.

    Expected layout:
        <repo_root>/
          data/
            LiveMathBench/
              CNMO_en.jsonl
              CCEE_en.jsonl
              AMC_en.jsonl
              WLPMC_en.jsonl
              hard_en.jsonl
    """
    # This file lives in: <repo_root>/src/experiment
    this_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    data_dir = os.path.join(repo_root, "data", "LiveMathBench")
    return data_dir


def load_livemathbench(
    split: str = "CNMO",
    n_examples: Optional[int] = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Load a LiveMathBench English split from local JSONL files.

    split ∈ {"CNMO", "CCEE", "AMC", "WLPMC", "HARD"}
    """
    assert split in LMB_SPLIT_TO_FILE, f"Invalid split={split}. Valid: {list(LMB_SPLIT_TO_FILE.keys())}"

    data_dir = get_lmb_data_dir()
    filename = LMB_SPLIT_TO_FILE[split]
    path = os.path.join(data_dir, filename)

    print(f"[LiveMathBench] Loading split='{split}' from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Make sure you put the downloaded JSONL files into data/LiveMathBench/"
        )

    df = pd.read_json(path, lines=True)

    expected_cols = ["question", "answer"]
    for c in expected_cols:
        assert c in df.columns, f"Expected column '{c}' not found. Got columns: {df.columns.tolist()}"

    df = df[expected_cols].copy()
    total = len(df)
    print(f"[LiveMathBench] Loaded {total} rows for split='{split}'.")

    if n_examples is not None:
        n = min(n_examples, total)
        if n < n_examples:
            print(f"[LiveMathBench] Requested {n_examples}, but only {total} available → using {n}.")
        df = df.sample(n=n, random_state=random_state).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


# ============================================================
# 2. GPT-5 caller
# ============================================================

class GPT5Caller:
    def __init__(self, model_name: str = "gpt-5.1"):
        load_dotenv()
        self.client = OpenAI()
        self.model_name = model_name

    def call(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Simple wrapper. One call, no streaming.
        """
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=temperature,
        )
        return response.output[0].content[0].text


# ============================================================
# 3. Prompt builder: PoT-Strong for general math
# ============================================================

def build_pot_prompt_math(question: str) -> str:
    """
    Ask GPT-5.1 to solve a math problem by writing Python code.
    """
    return (
        "You are an expert competition mathematician and Python programmer.\n"
        "You will solve the following math problem by writing a Python function.\n\n"
        "Requirements:\n"
        "1. Write a Python function with signature `def solve():` that returns the final answer.\n"
        "2. You may use only Python's built-in libraries and the `math` and `fractions` modules.\n"
        "   Do NOT use external libraries (no sympy, no numpy, etc.).\n"
        "3. Do not print anything. Just compute and return the final answer.\n"
        "4. The answer can be an integer, a float, or a rational number.\n"
        "   If the answer is a fraction, you may return it as a float or as a Python Fraction.\n"
        "5. Only output the Python code, inside a single code block starting with ```python and ending with ```.\n\n"
        "Math problem:\n"
        f"{question}\n"
    )


# ============================================================
# 4. PoT execution (strong)
# ============================================================

def extract_code_block(text: str) -> Optional[str]:
    """
    Extract the first ```python ... ``` block, or any ``` ... ``` block.
    """
    m = re.search(r"```python(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def execute_pot_code_strong(code: str) -> Optional[Any]:
    """
    Execute model-generated code in an isolated namespace and call solve().
    We allow access to math and fractions modules.
    """
    local_ns: Dict[str, Any] = {}
    try:
        import math
        from fractions import Fraction

        global_ns = {
            "__builtins__": __builtins__,
            "math": math,
            "Fraction": Fraction,
        }
        exec(code, global_ns, local_ns)
        if "solve" not in local_ns and "solve" not in global_ns:
            return None
        solve_fn = local_ns.get("solve", global_ns.get("solve"))
        result = solve_fn()
        return result
    except Exception as e:
        print("[execute_pot_code_strong] Error executing code:", repr(e))
        return None


# ============================================================
# 5. Answer normalization & verification
# ============================================================

def _try_parse_float(s: str) -> Optional[float]:
    """
    Try to parse a string as float. Return None on failure.
    """
    try:
        s_norm = (
            str(s)
            .replace("$", "")
            .replace("\\(", "")
            .replace("\\)", "")
            .strip()
        )
        return float(s_norm)
    except Exception:
        return None


def normalize_answer_str(ans: Any) -> str:
    """
    Convert answer to a canonical string form for comparison.
    - Fraction -> 'p/q'
    - int/float -> normalized decimal
    - else: lowercase, no spaces, strip simple LaTeX markers
    """
    from fractions import Fraction

    if isinstance(ans, Fraction):
        return f"{ans.numerator}/{ans.denominator}"
    if isinstance(ans, (int, float)):
        return ("{:.12g}".format(float(ans))).strip()

    s = str(ans)
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "")
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    return s.lower()


def verify_math_answer(pred: Any, gold: str, tol: float = 1e-6) -> Dict[str, Any]:
    """
    Compare model prediction (numeric / Fraction / string) with gold answer.
    1) Try numeric comparison if possible.
    2) Also compare normalized string forms.
    """
    if pred is None:
        return {
            "correct": False,
            "numeric_match": False,
            "string_match": False,
            "pred_norm": None,
            "gold_norm": normalize_answer_str(gold),
        }

    gold_float = _try_parse_float(gold)
    if isinstance(pred, (int, float)):
        pred_float = float(pred)
    else:
        pred_float = _try_parse_float(str(pred))

    numeric_match = False
    if gold_float is not None and pred_float is not None:
        if abs(pred_float - gold_float) <= tol * max(1.0, abs(gold_float)):
            numeric_match = True

    pred_norm = normalize_answer_str(pred)
    gold_norm = normalize_answer_str(gold)
    string_match = (pred_norm == gold_norm)

    return {
        "correct": numeric_match or string_match,
        "numeric_match": numeric_match,
        "string_match": string_match,
        "pred_norm": pred_norm,
        "gold_norm": gold_norm,
    }


# ============================================================
# 6. Main evaluation loop (PoT-Strong on any split)
# ============================================================

def run_livemathbench_pot_strong(
    split: str = "CNMO",
    n_examples: Optional[int] = None,
    random_state: int = 0,
    tol: float = 1e-6,
) -> None:
    """
    Run GPT-5.1 PoT-Strong on a LiveMathBench split.
      split ∈ {"CNMO", "CCEE", "AMC", "WLPMC", "HARD"}
    """
    df = load_livemathbench(
        split=split,
        n_examples=n_examples,
        random_state=random_state,
    )
    caller = GPT5Caller(model_name="gpt-5.1")

    results: List[Dict[str, Any]] = []

    total = len(df)
    print(f"\n=== Evaluating LiveMathBench split='{split}' with PoT-Strong (n={total}) ===\n")

    for i, row in df.iterrows():
        q = row["question"]
        gold = row["answer"]

        print(f"\n=== Example {i+1}/{total} (split={split}) ===")
        print(q[:200] + ("..." if len(q) > 200 else ""))
        print("Gold answer (raw):", gold)

        prompt = build_pot_prompt_math(q)
        t0 = time.time()
        try:
            raw_text = caller.call(prompt, temperature=0.0)
        except Exception as e:
            latency = time.time() - t0
            print("[ERROR] OpenAI call failed:", repr(e))
            results.append({
                "idx": i,
                "split": split,
                "question": q,
                "gold_answer": gold,
                "raw_response": None,
                "code": None,
                "pred_raw": None,
                "correct": False,
                "numeric_match": False,
                "string_match": False,
                "pred_norm": None,
                "gold_norm": normalize_answer_str(gold),
                "latency_sec": latency,
                "error": repr(e),
            })
            continue

        latency = time.time() - t0
        code = extract_code_block(raw_text)
        if code is None:
            print("[WARN] No code block found.")
            pred_val = None
        else:
            pred_val = execute_pot_code_strong(code)

        verif = verify_math_answer(pred_val, gold, tol=tol)

        print("Raw response (first 200 chars):")
        print((raw_text or "")[:200].replace("\n", " ") + ("..." if raw_text and len(raw_text) > 200 else ""))
        print("Predicted value:", pred_val)
        print("Verification:", verif)

        results.append({
            "idx": i,
            "split": split,
            "question": q,
            "gold_answer": gold,
            "raw_response": raw_text,
            "code": code,
            "pred_raw": pred_val,
            "correct": verif["correct"],
            "numeric_match": verif["numeric_match"],
            "string_match": verif["string_match"],
            "pred_norm": verif["pred_norm"],
            "gold_norm": verif["gold_norm"],
            "latency_sec": latency,
            "error": None,
        })

    res_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    suffix = "full" if n_examples is None else f"n{len(res_df)}"
    out_path = f"results/livemathbench_{split.lower()}_gpt5_pot_strong_{suffix}.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to: {out_path}")

    n = len(res_df)
    n_ok = int(res_df["correct"].sum())
    acc = n_ok / n if n > 0 else 0.0
    mean_latency = res_df["latency_sec"].mean()

    print(f"\n=== Summary (LiveMathBench {split}, PoT-Strong) ===")
    print(f"n = {n}")
    print(f"n_ok = {n_ok}")
    print(f"acc = {acc}")
    print(f"mean_latency_sec = {mean_latency}")


if __name__ == "__main__":
    # Example: run on 20 CNMO problems
    run_livemathbench_pot_strong(
        split="CNMO",   # change to "AMC", "CCEE", "WLPMC", "HARD"
        n_examples=None,  # set to None for full split
        random_state=0,
        tol=1e-6,
    )
