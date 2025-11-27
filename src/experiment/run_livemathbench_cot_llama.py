import os
import time
from typing import Any, Dict, Optional, List

import pandas as pd

from data_loader_livemathbench import load_livemathbench
from model.lama3_3b_loader import load_llama3_3b
from utils.hf_local_caller import HFLocalCaller
import re
from fractions import Fraction

def _try_parse_float(s: str):
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

def normalize_answer_str(ans):
    if isinstance(ans, Fraction):
        return f"{ans.numerator}/{ans.denominator}"
    if isinstance(ans, (int, float)):
        return ("{:.12g}".format(ans)).strip()

    s = str(ans)
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "")
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    return s.lower()

def verify_math_answer(pred, gold: str, tol: float = 1e-6):
    if pred is None:
        return {
            "correct": False,
            "numeric_match": False,
            "string_match": False,
            "pred_norm": None,
            "gold_norm": normalize_answer_str(gold),
        }

    gold_float = _try_parse_float(gold)
    pred_float = _try_parse_float(pred)

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

# ^ we already wrote these utilities in the analysis script


# ---------- 1. Prompt template (same as GPT-5 CoT) ----------
def build_cot_prompt(question: str) -> str:
    """
    Simple CoT-style prompt for Llama:
    - Explain the reasoning.
    - End with a single line: 'Answer: <number or expression>'
    """
    return (
        "You are an expert competition mathematician.\n"
        "Solve the following problem step by step with clear reasoning.\n"
        "At the very end, on the LAST line, write exactly:\n"
        "Answer: <final numeric answer>\n"
        "Replace <final numeric answer> with a single number or simple expression.\n"
        "Do NOT leave the placeholder text.\n"
        "Do NOT add any text after the answer on that line.\n\n"
        f"Problem:\n{question}\n"
    )






# ---------- 2. Small helper to extract the 'Answer:' line ----------

def extract_answer_from_cot(text: str) -> Optional[str]:
    if text is None:
        return None
    marker = "Answer:"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    ans = text[idx + len(marker):].strip()
    # Remove trailing junk newlines / sentences
    ans = ans.split("\n")[0].strip()
    return ans or None


# ---------- 3. Main runner ----------

def run_livemathbench_cot_llama(
    split: str = "all",
    n_examples: Optional[int] = None,
    random_state: int = 0,
    tol: float = 1e-6,
) -> None:
    # Load data (local JSONL files – same as GPT-5 script)
    df = load_livemathbench(
        split=split,
        n_examples=n_examples,
        random_state=random_state,
    )

    # Load Llama-3.2-3B and wrap in HFLocalCaller
    tokenizer, model = load_llama3_3b(device="cuda")
    #tokenizer, model = load_llama3_3b()
    caller = HFLocalCaller(tokenizer, model, max_new_tokens=256)

    results: List[Dict[str, Any]] = []
    total = len(df)
    print(f"\n=== Evaluating LiveMathBench [{split}] with CoT (Llama-3.2-3B, n={total}) ===\n")

    for i, row in df.iterrows():
        q = row["question"]
        gold = row["answer"]

        print(f"\n=== Example {i+1}/{total} ===")
        print(q[:200] + ("..." if len(q) > 200 else ""))
        print("Gold answer (raw):", gold)

        prompt = build_cot_prompt(q)
        t0 = time.time()
        try:
            raw_text = caller.call(prompt, temperature=0.2)
        except Exception as e:
            latency = time.time() - t0
            print("[ERROR] HF call failed:", repr(e))
            results.append({
                "idx": i,
                "split": split,
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
        print((raw_text or "")[:200].replace("\n", " ") +
              ("..." if raw_text and len(raw_text) > 200 else ""))
        print("Extracted ANSWER:", pred_str)
        print("Verification:", verif)

        results.append({
            "idx": i,
            "split": split,
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
    out_path = f"results/livemathbench_{split}_llama_cot.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to: {out_path}")

    # Summary
    n = len(res_df)
    n_ok = int(res_df["correct"].sum())
    acc = n_ok / n if n > 0 else 0.0
    mean_latency = res_df["latency_sec"].mean()

    print("\n=== Summary (LiveMathBench, CoT, Llama-3.2-3B) ===")
    print(f"split            = {split}")
    print(f"n                = {n}")
    print(f"n_ok             = {n_ok}")
    print(f"acc              = {acc}")
    print(f"mean_latency_sec = {mean_latency}")


if __name__ == "__main__":
    # Same default as GPT-5 script: all splits, all examples
    run_livemathbench_cot_llama(
        split="all",
        n_examples=None,
        random_state=0,
        tol=1e-6,
    )
