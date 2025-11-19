import os
import time
import pandas as pd

from nl4opt_utils import (
    load_nl4opt,
    GPT5Caller,
    build_pot_prompt_strong,
    extract_code_block,
    execute_pot_code_strong,
    verify_answer_numeric,
    summarize_results,
)


def run_gpt5_pot_strong(n_examples=None, random_state=0, tol=1e-3):
    print("=== Strong PoT (GPT-5.1) on NL4OPT ===")

    df = load_nl4opt(n_examples=n_examples, random_state=random_state)

    import pandas as pd
    df["en_answer_numeric"] = pd.to_numeric(df["en_answer"], errors="coerce")
    df = df[df["en_answer_numeric"].notna()].reset_index(drop=True)

    print(f"[clean] Using {len(df)} numeric examples.")

    caller = GPT5Caller(model_name="gpt-5.1")

    rows = []
    for i, row in df.iterrows():
        q = row["en_question"]
        gt = float(row["en_answer_numeric"])

        print(f"\n=== Example {i+1}/{len(df)} ===")
        print(q[:160] + ("..." if len(q) > 160 else ""))

        t0 = time.time()

        try:
            prompt = build_pot_prompt_strong(q)
            raw = caller.call(prompt, temperature=0.0)

            code = extract_code_block(raw)
            if code is None:
                pred = None
            else:
                pred = execute_pot_code_strong(code)

            verif = verify_answer_numeric(pred, gt, tol=tol)
            latency = time.time() - t0

            print(f"Pred={pred}, correct={verif['correct']}, abs_err={verif['abs_error']}")

            rows.append({
                "example": i,
                "gt": gt,
                "pred": pred,
                "correct": verif["correct"],
                "abs_err": verif["abs_error"],
                "rel_err": verif["rel_error"],
                "latency": latency,
            })

        except Exception as e:
            print("Error:", e)
            rows.append({
                "example": i,
                "gt": gt,
                "pred": None,
                "correct": False,
                "abs_err": None,
                "rel_err": None,
                "latency": None,
                "error": repr(e),
            })

    dfres = pd.DataFrame(rows)

    os.makedirs("results", exist_ok=True)
    out = "results/nl4opt_gpt5_pot_strong.csv" if n_examples is None \
          else "results/nl4opt_gpt5_pot_strong_sample.csv"

    dfres.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    summary = summarize_results(dfres)
    print("\n=== Summary (Strong PoT) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run_gpt5_pot_strong(
        n_examples=None,
        random_state=0,
        tol=1e-3
    )
