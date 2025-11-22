#!/usr/bin/env python3
"""
analyze_nl4opt_pot_failures.py

Goal:
- Compare baseline PoT vs strong PoT on NL4OPT.
- Classify examples into:
    * fixed:    baseline wrong, strong correct
    * regressed: baseline correct, strong wrong
    * both_wrong: both wrong
- For a few examples in each group, print:
    * question text
    * baseline PoT prediction + extracted code
    * strong PoT prediction
    * (optional) other strategies' performance from nl4opt_gpt5_strategies.csv
"""

import re
from textwrap import indent

import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def load_nl4opt_numeric() -> pd.DataFrame:
    """
    Load NL4OPT, keep only rows with numeric en_answer.
    Make sure indexing aligns with your PoT runs.
    """
    path = "hf://datasets/CardinalOperations/NL4OPT/NL4OPT_with_optimal_solution.json"
    print(f"[load_nl4opt_numeric] Loading from {path}")
    df = pd.read_json(path, lines=True)

    # Same cleaning as in your eval scripts
    df["en_answer_numeric"] = pd.to_numeric(df["en_answer"], errors="coerce")
    df = df[df["en_answer_numeric"].notna()].reset_index(drop=True)
    print(f"[load_nl4opt_numeric] Using {len(df)} numeric examples.")
    return df


def extract_code_block(text: str) -> str:
    """
    Extract the first ```python ... ``` or ``` ... ``` block from raw_response.
    If none found, return the whole text truncated.
    """
    if not isinstance(text, str):
        return ""
    m = re.search(r"```python(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if not m:
        # No code block, just return a short snippet of the whole answer
        return text.strip()
    return m.group(1).strip()


def short(text: str, max_len: int = 400) -> str:
    """
    Single-line, truncated version of text.
    """
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())
    return text[:max_len] + ("..." if len(text) > max_len else "")


def print_example_header(group_name: str, idx: int):
    print("=" * 80)
    print(f"[{group_name.upper()}] Example index = {idx}")
    print("=" * 80)


def print_other_strategies(df_strat: pd.DataFrame, idx: int):
    """
    For the 10-example multi-strategy run, show how each strategy did
    on this example index (if present).
    """
    if df_strat is None:
        return

    sub = df_strat[df_strat["example_idx"] == idx]
    if sub.empty:
        return

    print("\n[Multi-strategy performance on this example (10-example run)]")
    for _, r in sub.sort_values("strategy").iterrows():
        print(
            f"  - {r['strategy']:12s} | "
            f"correct={r['correct']} | "
            f"parsed={r['parsed_answer']} | "
            f"abs_err={r['abs_error']}"
        )
    print()


# -----------------------------
# Main analysis
# -----------------------------

def main(
    n_examples_per_group: int = 3,
    show_code: bool = True,
):
    # 1) Load data
    df_data = load_nl4opt_numeric()

    df_pot = pd.read_csv("results/nl4opt_gpt5_pot_full.csv")
    df_strong = pd.read_csv("results/nl4opt_gpt5_pot_strong.csv")

    try:
        df_strat = pd.read_csv("results/nl4opt_gpt5_strategies.csv")
    except FileNotFoundError:
        df_strat = None
        print("[warn] nl4opt_gpt5_strategies.csv not found; skipping multi-strategy view.")

    # 2) Align indexes
    df_pot = df_pot.set_index("example_idx")
    df_strong = df_strong.set_index("example")

    common_idx = sorted(set(df_pot.index) & set(df_strong.index))
    df_pot = df_pot.loc[common_idx]
    df_strong = df_strong.loc[common_idx]

    # 3) Define groups
    baseline_ok = df_pot["correct"].astype(bool)
    strong_ok = df_strong["correct"].astype(bool)

    fixed_mask = (~baseline_ok) & strong_ok
    regressed_mask = baseline_ok & (~strong_ok)
    both_wrong_mask = (~baseline_ok) & (~strong_ok)

    print("\n=== Global stats (PoT vs Strong PoT) ===")
    print(f"Total common examples: {len(common_idx)}")
    print(f"  fixed (baseline wrong -> strong correct): {fixed_mask.sum()}")
    print(f"  regressed (baseline correct -> strong wrong): {regressed_mask.sum()}")
    print(f"  both wrong: {both_wrong_mask.sum()}")

    # 4) Helper to print representative examples
    def show_group(group_name: str, mask):
        idxs = df_pot[mask].index.tolist()
        if not idxs:
            print(f"\n[{group_name}] No examples.")
            return
        print(f"\n[{group_name}] Showing up to {n_examples_per_group} examples...")
        for idx in idxs[:n_examples_per_group]:
            print_example_header(group_name, idx)

            # Core info
            q = df_data.loc[idx, "en_question"]
            gt = df_data.loc[idx, "en_answer_numeric"]

            pot_row = df_pot.loc[idx]
            strong_row = df_strong.loc[idx]

            print("Question:")
            print(indent(q.strip(), "  "))
            print(f"\nGround truth objective: {gt}")

            print("\nBaseline PoT:")
            print(f"  parsed_answer = {pot_row['parsed_answer']}")
            print(f"  correct       = {pot_row['correct']}")
            print(f"  abs_error     = {pot_row['abs_error']}")
            if show_code:
                code = extract_code_block(pot_row.get("raw_response", ""))
                print("\n  Baseline PoT code snippet:")
                print(indent(code, "    "))

            print("\nStrong PoT:")
            print(f"  pred          = {strong_row['pred']}")
            print(f"  correct       = {strong_row['correct']}")
            print(f"  abs_error     = {strong_row['abs_error']}")
            print(f"  rel_error     = {strong_row['rel_error']}")
            print(f"  latency_sec   = {strong_row['latency_sec']}")

            # Optional: how the other strategies did on this example
            print_other_strategies(df_strat, idx)

    # 5) Print some examples from each group
    show_group("fixed", fixed_mask)
    show_group("regressed", regressed_mask)
    show_group("both_wrong", both_wrong_mask)


if __name__ == "__main__":
    main()
