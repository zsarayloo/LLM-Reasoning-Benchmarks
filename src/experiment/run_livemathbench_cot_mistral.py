# src/experiment/run_livemathbench_cot_mistral.py

import os
import re
import time
from typing import Any, Dict

import torch
import pandas as pd

from data_loader_livemathbench import load_livemathbench
from model.mistral_7b_loader import load_mistral_7b
from analyze_livemathbench_cot_vs_sc import normalize_answer_str, verify_math_answer
from utils.hf_local_caller import HfLocalCaller


# ----------------- Prompt -----------------

def build_cot_prompt(question: str) -> str:
    """
    CoT-style prompt for Mistral:
    - Explain step by step.
    - End with: 'Answer: <final numeric answer>'
    """
    return (
        "You are an expert competition mathematician.\n"
        "Solve the following problem step by step with clear reasoning.\n"
        "At the very end, on the LAST line, write exactly:\n"
        "Answer: <final numeric answer>\n"
        "Replace <final numeric answer> with a single number or simple expression.\n"
        "Do NOT leave the placeholder text.\n"
        "Do NOT add anything after the answer on that line.\n\n"
        f"Problem:\n{question}\n"
    )


# ----------------- Answer extraction -----------------

def extract_answer_from_text(text: str) -> str | None:
    """
    Find the last line starting with 'Answer:' and return what follows.
    """
    if text is None:
        return None
    # Take the last occurrence to be safe
    matches = list(re.finditer(r"Answer\s*:\s*(.+)", text, flags=re.IGNORECASE))
    if not matches:
        return None
    ans = matches[-1].group(1).strip()
    return ans or None


# ----------------- Main eval -----------------

def run_livemathbench_cot_mistral(
    split: str = "all",
    n_examples: int = 20,
    random_state: int = 0,
    max_new_tokens: int = 256,
    tol: float = 1e-6,
) -> None:
    # Load data & subsample
    df = load_livemathbench(split=split)
    if n_examples is not None and n_examples < len(df):
        df = df.sample(n=n_examples, random_state=random_state).reset_index(drop=True)
        print(f"[load_livemathbench] Using SAMPLE set: {len(df)} examples from split='{split}'")
    else:
        print(f"[load_livemathbench] Using FULL set: {len(df)} examples from split='{split}'")

    # Load Mistral
    tokenizer, model = load_mistral_7b()
    caller = HfLocalCaller(
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.95,
    )

    results: list[Dict[str, Any]] = []

    n = len(df)
    print(f"\n=== Evaluating LiveMathBench [{split}] with CoT (Mistral-7B, n={n}) ===\n")

    for i, row in df.iterrows():
        q = row["question"]
        gold = row["answer"]
        idx = row.get("idx", i)
        split_name = row.get("split", split)

        print(f"\n=== Example {i+1}/{n} (idx={idx}, split={split_name}) ===")
        print(q[:200] + ("..." if len(q) > 200 else ""))
        print("Gold answer (raw):", gold)

        prompt = build_cot_prompt(q)
        t0 = time.time()

        try:
            raw_text = caller.call(prompt)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED by user]")
            break
        except Exception as e:
            latency = time.time() - t0
            print("[ERROR] Mistral call failed:", repr(e))
            results.append({
                "idx": idx,
                "split": split_name,
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

        print("Raw response (first 200 chars):")
        print((raw_text or "")[:200].replace("\n", " ") + ("..." if raw_text and len(raw_text) > 200 else ""))

        ans_str = extract_answer_from_text(raw_text)
        print("Extracted ANSWER:", ans_str)

        verif = verify_math_answer(ans_str, gold, tol=tol)
        print("Verification:", verif)

        results.append({
            "idx": idx,
            "split": split_name,
            "question": q,
            "gold_answer": gold,
            "raw_response": raw_text,
            "pred_str": ans_str,
            "correct": verif["correct"],
            "numeric_match": verif["numeric_match"],
            "string_match": verif["string_match"],
            "pred_norm": verif["pred_norm"],
            "gold_norm": verif["gold_norm"],
            "latency_sec": latency,
            "error": None,
        })

    # Save results
    res_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    out_path = f"results/livemathbench_{split}_mistral7b_cot_sample.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to: {out_path}")

    # Summary
    n = len(res_df)
    n_ok = int(res_df["correct"].sum())
    acc = n_ok / n if n > 0 else 0.0
    mean_latency = res_df["latency_sec"].mean()

    print("\n=== Summary (LiveMathBench, CoT, Mistral-7B) ===")
    print(f"split            = {split}")
    print(f"n                = {n}")
    print(f"n_ok             = {n_ok}")
    print(f"acc              = {acc}")
    print(f"mean_latency_sec = {mean_latency}")


if __name__ == "__main__":
    run_livemathbench_cot_mistral(
        split="all",
        n_examples=20,      # small, fast baseline
        random_state=0,
        max_new_tokens=256,
        tol=1e-6,
    )
