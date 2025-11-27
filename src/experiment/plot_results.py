# src/experiment/plot_results.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _results_dir() -> Path:
    # plot_results.py is in src/experiment → repo_root = parents[2]
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir


def _load_results(filename: str) -> pd.DataFrame:
    results_dir = _results_dir()
    path = results_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected results file not found: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 1) LiveMathBench – accuracy per split & method
# ---------------------------------------------------------------------------

def plot_livemathbench_method_summary():
    """
    Make a grouped bar chart: accuracy per competition split (AMC, CCEE, …)
    for GPT-5 CoT, GPT-5 CoT+SelfConsistency, LLaMA-3 CoT and Mistral-7B CoT.
    """
    results_dir = _results_dir()

    df_cot = _load_results("livemathbench_all_gpt5_cot.csv")
    df_sc = _load_results("livemathbench_all_gpt5_cot_selfconsistency.csv")
    df_llama = _load_results("livemathbench_all_llama_cot.csv")
    df_mistral = _load_results("livemathbench_all_mistral7b_cot_sample.csv")

    # Helper: group by split and compute accuracy + mean latency
    def summarize(df: pd.DataFrame, acc_col: str, lat_col: str, label: str):
        df = df.copy()
        # Normalize split names: AMC vs AMC_en, etc.
        df["split_norm"] = df["split"].astype(str).str.replace(r"_en$", "", regex=True)
        grouped = (
            df.groupby("split_norm")
              .agg(accuracy=(acc_col, "mean"),
                   mean_latency_sec=(lat_col, "mean"))
              .reset_index()
        )
        grouped["method"] = label
        return grouped

    summaries = [
        summarize(df_cot, "correct", "latency_sec", "GPT-5 CoT"),
        summarize(df_sc, "sc_correct", "mean_latency_sec", "GPT-5 CoT + SelfConsistency"),
        summarize(df_llama, "correct", "latency_sec", "LLaMA-3 CoT"),
        summarize(df_mistral, "correct", "latency_sec", "Mistral-7B CoT (sample)"),
    ]
    summary = pd.concat(summaries, ignore_index=True)

    # Nice ordering of splits
    split_order = ["AMC", "CCEE", "CNMO", "WLPMC", "hard"]
    summary["split_norm"] = pd.Categorical(summary["split_norm"],
                                           categories=split_order,
                                           ordered=True)
    summary = summary.sort_values(["split_norm", "method"])

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_colors=summary["method"].nunique())
    sns.barplot(
        data=summary,
        x="split_norm",
        y="accuracy",
        hue="method",
        palette=palette,
    )

    plt.ylim(0, 1.05)
    plt.xlabel("Competition split", fontsize=12)
    plt.ylabel("Accuracy (fraction correct)", fontsize=12)
    plt.title("LiveMathBench – accuracy per split and method", fontsize=14)
    plt.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()

    out_path = results_dir / "livemathbench_method_accuracy_highres.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot_results] Saved LiveMathBench summary to {out_path}")


# ---------------------------------------------------------------------------
# 2) NL4OPT – method summary (accuracy + latency)
# ---------------------------------------------------------------------------

def plot_nl4opt_method_summary():
    """
    Horizontal bar charts for NL4OPT:
    - left: exact-match accuracy
    - right: mean latency per instance
    for all GPT-5 pipelines (LP variants + PoT variants).
    """
    results_dir = _results_dir()

    files = {
        "LP Gurobi": "nl4opt_gpt5_lp_gurobi_full.csv",
        "LP SelfCheck": "nl4opt_gpt5_lp_gurobi_selfcheck_full.csv",
        "LP SelfCheck + Patch": "nl4opt_gpt5_lp_gurobi_selfcheck_patch_full.csv",
        "LP Semantic": "nl4opt_gpt5_lp_gurobi_semantic_full.csv",
        "LP Semantic + Verifier": "nl4opt_gpt5_lp_gurobi_semantic_verifier_full.csv",
        "LP VerifierLoop": "nl4opt_gpt5_lp_gurobi_verifierloop_full.csv",
        "PoT baseline": "nl4opt_gpt5_pot_full.csv",
        "Strong PoT": "nl4opt_gpt5_pot_strong.csv",
    }

    rows = []
    for method, fname in files.items():
        df = _load_results(fname)
        acc = float(df["correct"].mean())
        lat = float(df["latency_sec"].mean())
        rows.append({"method": method, "accuracy": acc, "mean_latency_sec": lat})

    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    palette = sns.color_palette("tab10", n_colors=len(summary))

    # Accuracy subplot
    sns.barplot(
        data=summary,
        y="method",
        x="accuracy",
        hue="method",          # map colors via hue
        dodge=False,
        ax=axes[0],
        palette=palette,
        order=summary["method"],
        legend=False,
    )
    axes[0].set_xlim(0, 1.0)
    axes[0].set_xlabel("Accuracy (fraction of exact matches)", fontsize=11)
    axes[0].set_ylabel("")
    axes[0].set_title("NL4OPT – objective accuracy", fontsize=13)

    for p in axes[0].patches:
        width = p.get_width()
        axes[0].text(
            width + 0.01,
            p.get_y() + p.get_height() / 2,
            f"{width:.2f}",
            va="center",
            fontsize=9,
        )

    # Latency subplot
    sns.barplot(
        data=summary,
        y="method",
        x="mean_latency_sec",
        hue="method",
        dodge=False,
        ax=axes[1],
        palette=palette,
        order=summary["method"],
        legend=False,
    )
    axes[1].set_xlabel("Mean latency per instance (seconds)", fontsize=11)
    axes[1].set_ylabel("")
    axes[1].set_title("NL4OPT – average latency", fontsize=13)

    max_lat = summary["mean_latency_sec"].max()
    for p in axes[1].patches:
        width = p.get_width()
        axes[1].text(
            width + 0.02 * max_lat,
            p.get_y() + p.get_height() / 2,
            f"{width:.1f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    out_path = results_dir / "nl4opt_method_summary_highres.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot_results] Saved NL4OPT method summary to {out_path}")


# ---------------------------------------------------------------------------
# 3) NL4OPT – PoT vs Strong PoT by magnitude of optimal objective
# ---------------------------------------------------------------------------

def plot_pot_vs_strong_by_magnitude():
    """
    Compare plain PoT vs Strong PoT on NL4OPT as a function of |z*|
    (magnitude of the optimal objective).
    """
    results_dir = _results_dir()

    pot = _load_results("nl4opt_gpt5_pot_full.csv").rename(
        columns={"example_idx": "example", "correct": "correct_pot"}
    )
    strong = _load_results("nl4opt_gpt5_pot_strong.csv").rename(
        columns={"correct": "correct_strong"}
    )

    merged = pot[["example", "gt", "correct_pot"]].merge(
        strong[["example", "correct_strong"]],
        on="example",
        how="inner",
    )

    merged["mag"] = merged["gt"].abs()

    # Magnitude bins
    bins = [0, 10, 100, 1_000, 10_000, 100_000, 1e9]
    labels = ["0–10", "10–100", "100–1k", "1k–10k", "10k–100k", "≥100k"]
    merged["mag_bin"] = pd.cut(merged["mag"], bins=bins, labels=labels, right=False)

    stats = (
        merged.groupby("mag_bin", observed=False)
        .agg(
            pot_acc=("correct_pot", "mean"),
            strong_acc=("correct_strong", "mean"),
            n=("example", "size"),
        )
        .reset_index()
    )

    long = pd.melt(
        stats,
        id_vars=["mag_bin", "n"],
        value_vars=["pot_acc", "strong_acc"],
        var_name="method",
        value_name="accuracy",
    )
    long["method"] = long["method"].map(
        {"pot_acc": "PoT baseline", "strong_acc": "Strong PoT"}
    )

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set2", n_colors=2)
    sns.barplot(
        data=long,
        x="mag_bin",
        y="accuracy",
        hue="method",
        palette=palette,
    )

    plt.ylim(0, 1.05)
    plt.xlabel("Magnitude of optimal objective |z*|", fontsize=12)
    plt.ylabel("Accuracy (fraction correct)", fontsize=12)
    plt.title("NL4OPT – PoT vs Strong PoT by objective magnitude", fontsize=14)
    plt.legend(title="Method", loc="best")
    plt.tight_layout()

    out_path = results_dir / "nl4opt_pot_vs_strong_by_magnitude_highres.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot_results] Saved PoT vs Strong PoT magnitude plot to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sns.set_theme(style="whitegrid", context="talk")

    try:
        plot_livemathbench_method_summary()
    except Exception as e:
        print(f"[plot_results] Skipped LiveMathBench plot due to error: {e}")

    try:
        plot_nl4opt_method_summary()
    except Exception as e:
        print(f"[plot_results] Error while plotting NL4OPT method summary: {e}")

    try:
        plot_pot_vs_strong_by_magnitude()
    except Exception as e:
        print(f"[plot_results] Error while plotting PoT vs Strong PoT: {e}")


if __name__ == "__main__":
    main()
