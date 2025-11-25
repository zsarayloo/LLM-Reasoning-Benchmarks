"""
analyze_livemathbench_cot_vs_sc.py

Compare baseline CoT vs CoT+Self-Consistency on LiveMathBench,
do failure analysis, per-category breakdown, simple error-type clustering,
and generate a heatmap.

Usage:
    python src/experiment/analyze_livemathbench_cot_vs_sc.py
"""

import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    print(f"Loading: {path}")
    df = pd.read_csv(path)
    print("  Columns:", df.columns.tolist())
    print("  Shape:  ", df.shape)
    return df


def simple_error_cluster(question: str) -> str:
    """
    Very rough heuristic clustering by topic using the question text.
    Categories:
      - 'geometry'
      - 'number_theory'
      - 'algebra'
      - 'other'
    """
    if not isinstance(question, str):
        return "other"
    q = question.lower()

    # Geometry-ish signals
    geom_keywords = [
        "triangle", "circle", "angle", "square", "rectangle", "polygon",
        "perimeter", "area", "radius", "diameter", "circumference", "length",
        "height", "width"
    ]
    if any(k in q for k in geom_keywords):
        return "geometry"

    # Number theory-ish signals
    nt_keywords = [
        "prime", "gcd", "lcm", "divisible", "mod ", "modulo", "remainder",
        "congruent", "integer solutions"
    ]
    if any(k in q for k in nt_keywords):
        return "number_theory"

    # Algebra-ish signals
    alg_keywords = [
        "equation", "system", "solve for", "polynomial", "expression",
        "factor", "roots", "quadratic", "linear", "variable", "unknown"
    ]
    if any(k in q for k in alg_keywords):
        return "algebra"

    return "other"


def print_failure_examples(
    merged: pd.DataFrame,
    mask: pd.Series,
    title: str,
    max_n: int = 3
) -> None:
    subset = merged[mask]
    print(f"\n\n======== {title} (n={len(subset)}) ========")
    for _, row in subset.head(max_n).iterrows():
        print("\n--------------------------------------------------")
        print(f"idx: {row['idx']}")
        split_val = row.get("split", row.get("split_cot", "unknown"))
        print(f"split: {split_val}")

        q = row.get("question_cot", row.get("question_sc", ""))
        print("Question:")
        print(" ", q)

        print("Gold answer:", row.get("gold_answer_cot", row.get("gold_answer_sc", "")))

        print("\nCoT prediction (normalized):", row.get("pred_norm", None))
        print("CoT correct:", row.get("correct_cot", None))

        print("SC winning answer norm:", row.get("sc_winning_norm", None))
        print("SC correct:", row.get("sc_correct", None))

        rr = str(row.get("raw_response", ""))[:200].replace("\n", " ")
        print("\nRaw CoT response (first 200 chars):")
        print(" ", rr)


def summarize_global(merged: pd.DataFrame) -> None:
    """(A) Global summary tables for CoT vs CoT+SC."""
    print("\n=== (A) Global Summary ===")

    n = len(merged)
    n_cot_ok = int(merged["correct_cot"].sum())
    n_sc_ok = int(merged["sc_correct"].sum())
    acc_cot = n_cot_ok / n if n > 0 else 0.0
    acc_sc = n_sc_ok / n if n > 0 else 0.0

    n_fixed = int(((merged["correct_cot"] == False) & (merged["sc_correct"] == True)).sum())
    n_regressed = int(((merged["correct_cot"] == True) & (merged["sc_correct"] == False)).sum())
    n_both_wrong = int(((merged["correct_cot"] == False) & (merged["sc_correct"] == False)).sum())

    print(f"Total rows: {n}")
    print(f"  CoT correct: {n_cot_ok}  (acc = {acc_cot:.3f})")
    print(f"  SC  correct: {n_sc_ok}  (acc = {acc_sc:.3f})")
    print()
    print(f"  Fixed      (CoT wrong -> SC correct): {n_fixed}")
    print(f"  Regressed  (CoT correct -> SC wrong): {n_regressed}")
    print(f"  Both wrong (CoT wrong & SC wrong):    {n_both_wrong}")

    # Small summary table
    summary_rows = [
        {
            "metric": "CoT accuracy",
            "value": acc_cot,
            "n_correct": n_cot_ok,
        },
        {
            "metric": "CoT+SC accuracy",
            "value": acc_sc,
            "n_correct": n_sc_ok,
        },
        {
            "metric": "Fixed (CoT wrong -> SC correct)",
            "value": n_fixed / n if n > 0 else 0.0,
            "count": n_fixed,
        },
        {
            "metric": "Regressed (CoT correct -> SC wrong)",
            "value": n_regressed / n if n > 0 else 0.0,
            "count": n_regressed,
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    print("\nGlobal summary table:")
    print(summary_df.to_string(index=False))


def summarize_per_split(merged: pd.DataFrame) -> None:
    """(B) Per-category (split) analysis."""
    # Try to find the right split column
    if "split" in merged.columns:
        split_col = "split"
    elif "split_cot" in merged.columns:
        split_col = "split_cot"
    elif "split_sc" in merged.columns:
        split_col = "split_sc"
    else:
        print("\n=== (B) Per-category analysis: SKIPPED (no 'split' column) ===")
        return

    print(f"\n=== (B) Per-category analysis by '{split_col}' ===")
    grouped = (
        merged
        .groupby(split_col)
        .agg(
            n=("idx", "size"),
            n_cot_ok=("correct_cot", "sum"),
            n_sc_ok=("sc_correct", "sum"),
        )
        .reset_index()
    )
    grouped["acc_cot"] = grouped["n_cot_ok"] / grouped["n"]
    grouped["acc_sc"] = grouped["n_sc_ok"] / grouped["n"]
    grouped["delta_acc_sc_minus_cot"] = grouped["acc_sc"] - grouped["acc_cot"]

    print(grouped.to_string(index=False))


def summarize_by_error_cluster(merged: pd.DataFrame) -> None:
    """(C) Error-type clustering (geometry / number_theory / algebra / other)."""
    print("\n=== (C) Error-type clustering ===")

    # Choose a question column to build clusters from
    if "question_cot" in merged.columns:
        qcol = "question_cot"
    elif "question" in merged.columns:
        qcol = "question"
    elif "question_sc" in merged.columns:
        qcol = "question_sc"
    else:
        print("No question column found for clustering, skipping.")
        return

    merged["error_cluster"] = merged[qcol].apply(simple_error_cluster)

    grouped = (
        merged
        .groupby("error_cluster")
        .agg(
            n=("idx", "size"),
            n_cot_ok=("correct_cot", "sum"),
            n_sc_ok=("sc_correct", "sum"),
        )
        .reset_index()
    )
    grouped["acc_cot"] = grouped["n_cot_ok"] / grouped["n"]
    grouped["acc_sc"] = grouped["n_sc_ok"] / grouped["n"]
    grouped["delta_acc_sc_minus_cot"] = grouped["acc_sc"] - grouped["acc_cot"]

    print(grouped.to_string(index=False))


def make_heatmap(merged: pd.DataFrame, out_path: str = "results/livemathbench_cot_sc_heatmap.png") -> None:
    """
    (D) Self-consistency error heatmap:
    rows = split, cols = error_cluster, values = CoT and SC accuracy.

    We’ll produce two heatmaps stacked vertically: CoT acc and SC acc.
    """

    # Ensure clustering & split exist
    if "error_cluster" not in merged.columns:
        # If not already computed, compute now
        if "question_cot" in merged.columns:
            qcol = "question_cot"
        elif "question" in merged.columns:
            qcol = "question"
        elif "question_sc" in merged.columns:
            qcol = "question_sc"
        else:
            print("\n=== (D) Heatmap: SKIPPED (no question column) ===")
            return
        merged["error_cluster"] = merged[qcol].apply(simple_error_cluster)

    if "split" in merged.columns:
        split_col = "split"
    elif "split_cot" in merged.columns:
        split_col = "split_cot"
    elif "split_sc" in merged.columns:
        split_col = "split_sc"
    else:
        print("\n=== (D) Heatmap: SKIPPED (no split column) ===")
        return

    # Build summary table: per (split, error_cluster)
    grouped = (
        merged
        .groupby([split_col, "error_cluster"])
        .agg(
            n=("idx", "size"),
            n_cot_ok=("correct_cot", "sum"),
            n_sc_ok=("sc_correct", "sum"),
        )
        .reset_index()
    )
    grouped["acc_cot"] = grouped["n_cot_ok"] / grouped["n"]
    grouped["acc_sc"] = grouped["n_sc_ok"] / grouped["n"]

    # Pivot to matrices
    pivot_cot = grouped.pivot(index=split_col, columns="error_cluster", values="acc_cot")
    pivot_sc = grouped.pivot(index=split_col, columns="error_cluster", values="acc_sc")

    # Create heatmaps
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)

    for ax, pivot, title in [
        (axes[0], pivot_cot, "CoT accuracy"),
        (axes[1], pivot_sc, "CoT + Self-Consistency accuracy"),
    ]:
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isnan(val):
                    txt = "–"
                else:
                    txt = f"{val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("LiveMathBench: CoT vs CoT+Self-Consistency accuracy by split & error type")
    plt.savefig(out_path, dpi=200)
    print(f"\n=== (D) Heatmap saved to: {out_path} ===")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    path_cot = "results/livemathbench_all_gpt5_cot.csv"
    path_sc = "results/livemathbench_all_gpt5_cot_selfconsistency.csv"

    df_cot = load_csv(path_cot)
    df_sc = load_csv(path_sc)

    # Standardize some column names for clarity
    # CoT file: question, gold_answer, correct, ...
    if "question" in df_cot.columns:
        df_cot = df_cot.rename(columns={"question": "question_cot"})
    if "gold_answer" in df_cot.columns:
        df_cot = df_cot.rename(columns={"gold_answer": "gold_answer_cot"})
    if "correct" in df_cot.columns:
        df_cot = df_cot.rename(columns={"correct": "correct_cot"})

    # SC file: question_sc, gold_answer_sc, sc_correct, ...
    # (already named well in your run, so we don’t rename here)

    # Merge on idx only (split may differ in naming)
    merged = df_cot.merge(df_sc, on="idx", how="inner", suffixes=("_cot", "_sc"))
    print("Merged rows:", len(merged))

    # Sanity: ensure the two splits are the same if both exist
    if "split_cot" in merged.columns and "split_sc" in merged.columns:
        mismatch = (merged["split_cot"] != merged["split_sc"]).sum()
        if mismatch > 0:
            print(f"[WARN] {mismatch} rows have different split_cot vs split_sc")

        # For simplicity, create a unified split column
        merged["split"] = merged["split_cot"]

    # Check we have the expected correctness columns
    if "correct_cot" not in merged.columns:
        raise ValueError(
            "Could not find 'correct_cot' column after merging. "
            "Check the CSV structure."
        )
    if "sc_correct" not in merged.columns:
        raise ValueError(
            "Could not find 'sc_correct' column after merging. "
            "Check the self-consistency CSV structure."
        )

    # (A) Global summary
    summarize_global(merged)

    # Define failure group masks
    merged["fixed"] = (merged["correct_cot"] == False) & (merged["sc_correct"] == True)
    merged["regressed"] = (merged["correct_cot"] == True) & (merged["sc_correct"] == False)
    merged["both_wrong"] = (merged["correct_cot"] == False) & (merged["sc_correct"] == False)

    print("\n=== Failure Group Counts ===")
    print("fixed:     ", int(merged["fixed"].sum()))
    print("regressed: ", int(merged["regressed"].sum()))
    print("both_wrong:", int(merged["both_wrong"].sum()))

    # Print representative examples
    print_failure_examples(merged, merged["fixed"], "Fixed by Self-Consistency", max_n=3)
    print_failure_examples(merged, merged["regressed"], "Regressed due to Self-Consistency", max_n=3)
    print_failure_examples(merged, merged["both_wrong"], "Both CoT and SC wrong", max_n=3)

    # (B) Per-split summary
    summarize_per_split(merged)

    # (C) Error-type clustering
    summarize_by_error_cluster(merged)

    # (D) Heatmap
    make_heatmap(merged, out_path="results/livemathbench_cot_sc_heatmap.png")


if __name__ == "__main__":
    main()
