# src/experiment/run_nl4opt_pot_gpt5.py

import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from nl4opt_utils import (
    load_nl4opt,
    GPT5Caller,
    build_pot_prompt,
    extract_code_block,
    execute_pot_code,
    verify_answer_numeric,
    summarize_results,
)


def run_gpt5_pot_on_nl4opt(
    n_examples: Optional[int] = None,
    random_state: int = 0,
    tol: float = 1e-3,
    sleep_sec: float = 0.1,
):
    """
    Run GPT-5.1 Program-of-Thoughts on NL4OPT.

    n_examples = None  -> full dataset (245 examples).
    """
    print("=== run_nl4opt_pot_gpt5.py ===")
    print(f"n_examples={n_examples}, random_state={random_state}, tol={tol}")

    df = load_nl4opt(n_examples=n_examples, random_state=random_state)
    caller = GPT5Caller(model_name="gpt-5.1")

    results: List[Dict[str, Any]] = []

    total = len(df)
    print(f"\n[run] Total problems to solve with PoT: {total}\n")

    for i, row in df.iterrows():
        question = row["en_question"]
        gt = float(row["en_answer"])

        print(f"\n=== Example {i+1}/{total} ===")
        print("Question (truncated to 200 chars):")
        print(question[:200] + ("..." if len(question) > 200 else ""))
        print("Ground truth objective value:", gt)

        t0 = time.time()

        try:
            # 1) Build PoT prompt and call GPT-5.1
            prompt = build_pot_prompt(question)
            raw_text = caller.call(prompt, temperature=0.0)

            # 2) Extract and execute code
            code = extract_code_block(raw_text)
            if code is None:
                pred = None
            else:
                pred = execute_pot_code(code)

            latency = time.time() - t0

            # 3) Verify numerically (no leakage back to the model)
            verif = verify_answer_numeric(pred, gt, tol=tol)

            preview = (raw_text or "")[:200].replace("\n", " ")
            print("Response (truncated to 200 chars):")
            print(preview + ("..." if raw_text and len(raw_text) > 200 else ""))
            print(
                f"Parsed={pred}, correct={verif['correct']}, "
                f"abs_err={verif['abs_error']}"
            )

            results.append({
                "example_idx": i,
                "gt": gt,
                "raw_response": raw_text,
                "parsed_answer": pred,
                "correct": verif["correct"],
                "abs_error": verif["abs_error"],
                "rel_error": verif["rel_error"],
                "latency_sec": latency,
                "error": None,
            })

        except Exception as e:
            latency = time.time() - t0
            print("[run] Error:", repr(e))
            results.append({
                "example_idx": i,
                "gt": gt,
                "raw_response": None,
                "parsed_answer": None,
                "correct": False,
                "abs_error": None,
                "rel_error": None,
                "latency_sec": latency,
                "error": repr(e),
            })

        # Small sleep to be nice to the API (you can tune/remove)
        time.sleep(sleep_sec)

    # Save detailed results
    res_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    out_path = "results/nl4opt_gpt5_pot_full.csv" if n_examples is None else \
               f"results/nl4opt_gpt5_pot_{len(df)}examples.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\n[run] Saved detailed results to: {out_path}")

    # Print a small summary
    summary = summarize_results(res_df)
    print("\n=== Summary (GPT-5.1, PoT, NL4OPT) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    # For full dataset:
    run_gpt5_pot_on_nl4opt(
        n_examples=None,   # <-- FULL NL4OPT; change to 10/20 for quick tests
        random_state=0,
        tol=1e-3,
        sleep_sec=0.1,
    )
