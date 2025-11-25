#!/usr/bin/env python3
import os
import re
import time
from typing import Any, Dict, Optional, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. Data loader for LiveMathBench (local JSONL files)
# ============================================================

LMB_BASE_DIR = os.path.join("data", "LiveMathBench")

# Mapping from split name to local filename
LMB_SPLIT_FILES = {
    "CNMO": "CNMO_en.jsonl",
    "AMC": "AMC_en.jsonl",
    "CCEE": "CCEE_en.jsonl",
    "WLPMC": "WLPMC_en.jsonl",
    "hard": "hard_en.jsonl",
}


def load_livemathbench_local(
    split: str = "CNMO",
    n_examples: Optional[int] = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Load a LiveMathBench split from local JSONL files.

    Parameters
    ----------
    split : {"CNMO","AMC","CCEE","WLPMC","hard","all"}
        Which subset to load. If "all", concatenates all five.
    n_examples : int or None
        Optional sample size.
    random_state : int
        Random seed for sampling.

    Returns
    -------
    df : DataFrame with at least columns ["question", "answer"].
    """
    if split == "all":
        dfs = []
        for k, fname in LMB_SPLIT_FILES.items():
            path = os.path.join(LMB_BASE_DIR, fname)
            print(f"[load_livemathbench_local] Loading split={k} from {path}")
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            df_part = pd.read_json(path, lines=True)
            df_part["__split"] = k
            dfs.append(df_part)
        df = pd.concat(dfs, ignore_index=True)
    else:
        if split not in LMB_SPLIT_FILES:
            raise ValueError(f"Unknown split '{split}'. "
                             f"Use one of {list(LMB_SPLIT_FILES.keys()) + ['all']}.")
        fname = LMB_SPLIT_FILES[split]
        path = os.path.join(LMB_BASE_DIR, fname)
        print(f"[load_livemathbench_local] Loading split={split} from {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_json(path, lines=True)
        df["__split"] = split

    # Check required columns
    expected_cols = ["question", "answer"]
    for c in expected_cols:
        assert c in df.columns, f"Expected column '{c}' not found; got {df.columns.tolist()}"

    df = df[expected_cols + ["__split"]].copy()

    if n_examples is not None:
        n_examples = min(n_examples, len(df))
        df = df.sample(n=n_examples, random_state=random_state).reset_index(drop=True)
        print(f"[load_livemathbench_local] Using sample of {len(df)} examples.")
    else:
        df = df.reset_index(drop=True)
        print(f"[load_livemathbench_local] Using FULL dataset: {len(df)} examples.")

    return df


# ============================================================
# 2. GPT-5 caller
# ============================================================

class GPT5Caller:
    """
    Minimal wrapper around OpenAI Responses API.
    """

    def __init__(self, model_name: str = "gpt-5.1"):
        load_dotenv()
        self.client = OpenAI()
        self.model_name = model_name

    def call(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Single call, non-streaming CoT.
        """
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=temperature,
        )
        return response.output[0].content[0].text


# ============================================================
# 3. CoT prompt builder for general math
# ============================================================

def build_cot_prompt_math(question: str) -> str:
    """
    Ask GPT-5.1 to solve a math problem with step-by-step reasoning,
    and then give a final answer in a clear 'ANSWER: ...' format.
    """
    return (
        "You are an expert competition mathematician.\n"
        "Solve the following problem step by step.\n\n"
        "Requirements:\n"
        "1. Think carefully and show your reasoning.\n"
        "2. At the very end, on the LAST line, write the final answer in the form:\n"
        "   ANSWER: <final_answer>\n"
        "3. Do not add any extra text after the ANSWER line.\n\n"
        "Problem:\n"
        f"{question}\n"
    )


# ============================================================
# 4. Answer extraction & normalization
# ============================================================

def extract_answer_from_cot(text: str) -> Optional[str]:
    """
    Extract the 'ANSWER: ...' line from a CoT-style response.
    If none found, fall back to the last non-empty line.
    """
    if not text:
        return None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Search from bottom for 'ANSWER:'
    for ln in reversed(lines):
        m = re.search(r"ANSWER\s*:\s*(.+)", ln, flags=re.IGNORECASE)
        if m:
            ans = m.group(1).strip()
            return ans

    # Fallback: use last non-empty line
    if lines:
        return lines[-1]
    return None


def _try_parse_float(s: str) -> Optional[float]:
    """
    Try to parse a string as float. Return None on failure.
    Handles simple LaTeX-like wrappers.
    """
    try:
        s_norm = (
            s.replace("$", "")
             .replace("\\(", "")
             .replace("\\)", "")
             .replace("\\,", "")
             .strip()
        )
        # Sometimes answers are like "2$" or "$2$" already cleaned above
        return float(s_norm)
    except Exception:
        return None


def normalize_answer_str(ans: Any) -> str:
    """
    Convert answer to a canonical string form for string-based comparison.
    - Remove LaTeX dollar signs and parentheses.
    - Remove whitespace.
    - Lowercase.
    """
    s = str(ans)
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "")
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    return s.lower()


def verify_math_answer(pred_str: Optional[str], gold_str: str, tol: float = 1e-6) -> Dict[str, Any]:
    """
    Compare a predicted answer (string from CoT) with the gold answer.

    Strategy:
      1) Try numeric comparison if both parse as floats.
      2) Otherwise compare normalized strings.

    Returns dict with flags.
    """
    if pred_str is None:
        return {
            "correct": False,
            "numeric_match": False,
            "string_match": False,
            "pred_norm": None,
            "gold_norm": normalize_answer_str(gold_str),
        }

    gold_float = _try_parse_float(gold_str)
    pred_float = _try_parse_float(pred_str)

    numeric_match = False
    if gold_float is not None and pred_float is not None:
        if abs(pred_float - gold_float) <= tol * max(1.0, abs(gold_float)):
            numeric_match = True

    pred_norm = normalize_answer_str(pred_str)
    gold_norm = normalize_answer_str(gold_str)
    string_match = (pred_norm == gold_norm)

    return {
        "correct": numeric_match or string_match,
        "numeric_match": numeric_match,
        "string_match": string_match,
        "pred_norm": pred_norm,
        "gold_norm": gold_norm,
    }


# ============================================================
# 5. Main evaluation loop (CoT on LiveMathBench)
# ============================================================

def run_livemathbench_cot(
    split: str = "CNMO",
    n_examples: Optional[int] = None,
    random_state: int = 0,
    tol: float = 1e-6,
) -> None:
    """
    Run GPT-5.1 CoT on a LiveMathBench split.

    Parameters
    ----------
    split : {"CNMO","AMC","CCEE","WLPMC","hard","all"}
    n_examples : int or None
    """
    df = load_livemathbench_local(
        split=split,
        n_examples=n_examples,
        random_state=random_state,
    )
    caller = GPT5Caller(model_name="gpt-5.1")

    results: List[Dict[str, Any]] = []

    total = len(df)
    print(f"\n=== Evaluating LiveMathBench [{split}] with CoT (n={total}) ===\n")

    for i, row in df.iterrows():
        q = row["question"]
        gold = row["answer"]

        print(f"\n=== Example {i+1}/{total} ===")
        print(q[:200] + ("..." if len(q) > 200 else ""))
        print("Gold answer (raw):", gold)

        prompt = build_cot_prompt_math(q)
        t0 = time.time()
        try:
            raw_text = caller.call(prompt, temperature=0.0)
        except Exception as e:
            latency = time.time() - t0
            print("[ERROR] OpenAI call failed:", repr(e))
            results.append({
                "idx": i,
                "split": row.get("__split", split),
                "question": q,
                "gold_answer": gold,
                "raw_response": None,
                "pred_str": None,
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
        pred_str = extract_answer_from_cot(raw_text)
        verif = verify_math_answer(pred_str, gold, tol=tol)

        print("Raw response (first 200 chars):")
        print((raw_text or "")[:200].replace("\n", " ") + ("..." if raw_text and len(raw_text) > 200 else ""))
        print("Extracted ANSWER:", pred_str)
        print("Verification:", verif)

        results.append({
            "idx": i,
            "split": row.get("__split", split),
            "question": q,
            "gold_answer": gold,
            "raw_response": raw_text,
            "pred_str": pred_str,
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
    out_name = f"livemathbench_{split.lower()}_gpt5_cot.csv"
    out_path = os.path.join("results", out_name)
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to: {out_path}")

    # Summary
    n = len(res_df)
    n_ok = int(res_df["correct"].sum())
    acc = n_ok / n if n > 0 else 0.0
    mean_latency = res_df["latency_sec"].mean()

    print("\n=== Summary (LiveMathBench, CoT) ===")
    print(f"split            = {split}")
    print(f"n                = {n}")
    print(f"n_ok             = {n_ok}")
    print(f"acc              = {acc}")
    print(f"mean_latency_sec = {mean_latency}")


# ============================================================
# 6. Entry point
# ============================================================

if __name__ == "__main__":
    # Example configs:
    #   split="CNMO", n_examples=20   # quick sanity check
    #   split="CNMO", n_examples=None # full split
    #   split="all"                   # all five splits concatenated
    run_livemathbench_cot(
        split="all",      # change to "AMC", "CCEE", "WLPMC", "hard", or "all"
        n_examples=None,     # set to None for full split
        random_state=0,
        tol=1e-6,
    )
