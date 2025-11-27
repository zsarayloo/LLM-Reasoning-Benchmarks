# src/experiment/plot_all_benchmarks.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'serif'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2,
    'patch.linewidth': 1.2
})

# Set high contrast color palette
COLORS = {
    'primary': '#1f77b4',      # Strong blue
    'secondary': '#d62728',    # Strong red  
    'tertiary': '#2ca02c',     # Strong green
    'quaternary': '#ff7f0e',   # Strong orange
    'quinary': '#9467bd',      # Strong purple
    'senary': '#8c564b',       # Strong brown
    'septenary': '#e377c2',    # Strong pink
    'octonary': '#7f7f7f',     # Strong gray
    'grid': '#cccccc',         # Light gray for grids
    'text': '#000000'          # Black for text
}

# Set seaborn style with high contrast
sns.set_palette([COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], 
                COLORS['quaternary'], COLORS['quinary'], COLORS['senary']])


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def _load(path: Path) -> pd.DataFrame:
    """Small helper with a clearer error if a file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    return pd.read_csv(path)


def _clean_nl4opt(df: pd.DataFrame) -> pd.DataFrame:
    """Drop sentinel rows (gt <= -9999) for magnitude/latency plots."""
    if "gt" in df.columns:
        return df[df["gt"] > -9999].copy()
    return df.copy()


# ---------------------------------------------------------------------
# 1) NL4OPT – Accuracy vs magnitude (PoT baseline vs Strong PoT)
# ---------------------------------------------------------------------

def plot_nl4opt_accuracy_vs_magnitude() -> None:
    pot_path = RESULTS_DIR / "nl4opt_gpt5_pot_full.csv"
    strong_path = RESULTS_DIR / "nl4opt_gpt5_pot_strong.csv"

    pot = _clean_nl4opt(_load(pot_path))
    strong = _clean_nl4opt(_load(strong_path))

    bins = [0, 10, 100, 1_000, 10_000, np.inf]
    labels = ["0–10", "10–100", "100–1k", "1k–10k", ">10k"]

    pot["magnitude_bin"] = pd.cut(pot["gt"], bins=bins, labels=labels)
    strong["magnitude_bin"] = pd.cut(strong["gt"], bins=bins, labels=labels)

    pot_acc = pot.groupby("magnitude_bin", observed=False)["correct"].mean().reindex(labels)
    strong_acc = strong.groupby("magnitude_bin", observed=False)["correct"].mean().reindex(labels)

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, pot_acc.values, width, label="PoT baseline", color=COLORS['primary'])
    ax.bar(x + width / 2, strong_acc.values, width, label="Strong PoT", color=COLORS['secondary'])

    ax.set_title("NL4OPT: Accuracy vs Objective Magnitude")
    ax.set_xlabel("Magnitude of Optimal Objective")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for xi, v in zip(x, pot_acc.values):
        if pd.notna(v):
            ax.text(xi - width / 2, v + 0.02, f"{v:.1%}", ha="center", fontsize=8)
    for xi, v in zip(x, strong_acc.values):
        if pd.notna(v):
            ax.text(xi + width / 2, v + 0.02, f"{v:.1%}", ha="center", fontsize=8)

    output_path = FIGURES_DIR / "nl4opt_accuracy_vs_magnitude.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------
# 2) NL4OPT – Accuracy vs latency summary (two horizontal bar charts)
# ---------------------------------------------------------------------

def plot_nl4opt_accuracy_latency_summary() -> None:
    files = [
        ("nl4opt_gpt5_pot_full.csv", "PoT baseline"),
        ("nl4opt_gpt5_pot_strong.csv", "Strong PoT"),
        ("nl4opt_gpt5_lp_gurobi_full.csv", "LP Gurobi"),
    ]

    rows = []
    for fname, label in files:
        df = _clean_nl4opt(_load(RESULTS_DIR / fname))
        acc = df["correct"].mean()
        lat = df["latency_sec"].median()
        rows.append({"Method": label, "Accuracy": acc, "Median Latency (s)": lat})

    summary = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    sns.barplot(
        data=summary, x="Accuracy", y="Method", ax=axes[0], orient="h"
    )
    axes[0].set_title("NL4OPT: Accuracy by Method")
    axes[0].set_xlim(0, 1.0)
    axes[0].grid(axis="x", linestyle="--", alpha=0.3)
    for i, v in enumerate(summary["Accuracy"]):
        axes[0].text(v + 0.01, i, f"{v:.1%}", va="center")

    sns.barplot(
        data=summary, x="Median Latency (s)", y="Method", ax=axes[1], orient="h"
    )
    axes[1].set_title("NL4OPT: Median Latency by Method")
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)
    for i, v in enumerate(summary["Median Latency (s)"]):
        axes[1].text(v + 0.02, i, f"{v:.2f}s", va="center")

    fig.suptitle("NL4OPT: Accuracy–Latency Summary", y=1.02, fontsize=12)
    plt.tight_layout()
    output_path = FIGURES_DIR / "nl4opt_accuracy_latency_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------
# 3) NL4OPT – Strong PoT diagnostics panel
# ---------------------------------------------------------------------

def plot_nl4opt_strong_pot_diagnostics() -> None:
    strong = _clean_nl4opt(_load(RESULTS_DIR / "nl4opt_gpt5_pot_strong.csv"))
    gurobi = _clean_nl4opt(_load(RESULTS_DIR / "nl4opt_gpt5_lp_gurobi_full.csv"))

    # Align on example index
    if "example" in strong.columns:
        strong = strong.rename(columns={"example": "example_idx"})
    if "example" in gurobi.columns:
        gurobi = gurobi.rename(columns={"example": "example_idx"})

    merged = pd.merge(
        strong[["example_idx", "gt", "pred", "correct", "latency_sec"]],
        gurobi[["example_idx", "correct", "pred_obj"]],
        on="example_idx",
        suffixes=("_strong", "_gurobi"),
    )

    # Magnitude bins
    bins = [0, 10, 100, 1_000, 10_000, np.inf]
    labels = ["0–10", "10–100", "100–1k", "1k–10k", ">10k"]
    merged["magnitude_bin"] = pd.cut(merged["gt"], bins=bins, labels=labels)
    acc_by_mag = (
        merged.groupby("magnitude_bin", observed=False)["correct_strong"].mean().reindex(labels)
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (1) Accuracy by magnitude
    ax = axes[0, 0]
    sns.barplot(
        x=acc_by_mag.index,
        y=acc_by_mag.values,
        ax=ax,
    )
    ax.set_title("Strong PoT: Accuracy by Problem Magnitude")
    ax.set_xlabel("Magnitude of Optimal Objective")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for i, v in enumerate(acc_by_mag.values):
        if pd.notna(v):
            ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=8)

    # (2) Log–log scatter GT vs prediction
    ax = axes[0, 1]
    plot_df = merged[(merged["gt"] > 0) & (merged["pred"] > 0)].copy()
    correct = plot_df[plot_df["correct_strong"]]
    wrong = plot_df[~plot_df["correct_strong"]]

    ax.plot(
        [plot_df["gt"].min(), plot_df["gt"].max()],
        [plot_df["gt"].min(), plot_df["gt"].max()],
        "k--",
        alpha=0.5,
        label="Ideal",
    )
    ax.scatter(correct["gt"], correct["pred"], alpha=0.7, label="Correct", color=COLORS['tertiary'], s=30)
    ax.scatter(wrong["gt"], wrong["pred"], alpha=0.8, marker="x", label="Incorrect", color=COLORS['secondary'], s=40)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")
    ax.set_title("Strong PoT: Prediction vs Ground Truth (Log–Log)")
    ax.legend()

    # (3) Latency boxplot
    ax = axes[1, 0]
    sns.boxplot(
        data=merged,
        x="correct_strong",
        y="latency_sec",
        ax=ax,
    )
    ax.set_xticklabels(["Wrong", "Correct"])
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Strong PoT: Latency vs Correctness")

    # (4) PoT vs Gurobi overlap pie
    def category(row):
        if row["correct_strong"] and row["correct_gurobi"]:
            return "Both Correct"
        if row["correct_strong"] and not row["correct_gurobi"]:
            return "Only PoT Correct"
        if not row["correct_strong"] and row["correct_gurobi"]:
            return "Only Gurobi Correct"
        return "Both Wrong"

    merged["overlap"] = merged.apply(category, axis=1)
    counts = merged["overlap"].value_counts()
    ax = axes[1, 1]
    colors_pie = [COLORS['tertiary'], COLORS['secondary'], COLORS['quaternary'], COLORS['primary']]
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors_pie[:len(counts)],
        textprops={'fontsize': 10, 'color': COLORS['text']}
    )
    ax.set_title("Strong PoT vs Gurobi: Overlap of Success")

    fig.suptitle("NL4OPT: Strong PoT Diagnostics", y=1.02, fontsize=12)
    plt.tight_layout()
    output_path = FIGURES_DIR / "nl4opt_strong_pot_diagnostics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------
# 4) LiveMathBench – Per-split accuracy bar chart
# ---------------------------------------------------------------------

def plot_lmb_per_split_accuracy() -> None:
    cot = _load(RESULTS_DIR / "livemathbench_all_gpt5_cot.csv")
    sc = _load(RESULTS_DIR / "livemathbench_all_gpt5_cot_selfconsistency.csv")

    cot["split_norm"] = cot["split"].str.replace("_en", "", regex=False)
    sc["split_norm"] = sc["split"].str.replace("_en", "", regex=False)

    splits = ["AMC", "CCEE", "CNMO", "WLPMC", "hard"]

    rows = []
    for split in splits:
        acc_cot = cot[cot["split_norm"] == split]["correct"].mean()
        acc_sc = sc[sc["split_norm"] == split]["sc_correct"].mean()
        rows.append({"Split": split, "Method": "CoT", "Accuracy": acc_cot})
        rows.append({"Split": split, "Method": "CoT + Self-Consistency", "Accuracy": acc_sc})

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x="Split", y="Accuracy", hue="Method", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_title("LiveMathBench: Per-Split Accuracy (GPT-5)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3)

    output_path = FIGURES_DIR / "livemathbench_per_split_accuracy.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------
# 5) LiveMathBench – Overall leaderboard
# ---------------------------------------------------------------------

def plot_lmb_leaderboard() -> None:
    sc = _load(RESULTS_DIR / "livemathbench_all_gpt5_cot_selfconsistency.csv")
    cot = _load(RESULTS_DIR / "livemathbench_all_gpt5_cot.csv")
    llama = _load(RESULTS_DIR / "livemathbench_all_llama_cot.csv")
    mistral = _load(RESULTS_DIR / "livemathbench_all_mistral7b_cot_sample.csv")

    rows = [
        {"Model": "GPT-5 (CoT + SC)", "Accuracy": sc["sc_correct"].mean()},
        {"Model": "GPT-5 (CoT)", "Accuracy": cot["correct"].mean()},
        {"Model": "Mistral 7B (CoT)", "Accuracy": mistral["correct"].mean()},
        {"Model": "LLaMA (CoT)", "Accuracy": llama["correct"].mean()},
    ]
    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x="Accuracy", y="Model", ax=ax, orient="h")
    ax.set_xlim(0, 1.0)
    ax.set_title("LiveMathBench: Overall Model Leaderboard")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    for i, v in enumerate(df["Accuracy"]):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center")

    output_path = FIGURES_DIR / "livemathbench_leaderboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------
# 6) LiveMathBench – CoT vs CoT+SC heatmap (use existing PNG)
# ---------------------------------------------------------------------

def plot_lmb_cot_sc_heatmap() -> None:
    img_path = RESULTS_DIR / "livemathbench_cot_sc_heatmap.png"
    if not img_path.exists():
        # If missing, just skip quietly.
        return
    img = plt.imread(img_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("LiveMathBench: CoT vs CoT+SC (Split × Topic)")
    plt.tight_layout()
    output_path = FIGURES_DIR / "livemathbench_cot_sc_heatmap_display.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------
# 7) PoT gap cross-dataset chart (NL4OPT vs CNMO, CoT vs PoT Strong)
# ---------------------------------------------------------------------

def plot_pot_gap_cross_dataset() -> None:
    # NL4OPT CoT accuracy from strategies sample (strategy == 'cot')
    strat = _load(RESULTS_DIR / "nl4opt_gpt5_strategies.csv")
    cot_nl4opt = strat[strat["strategy"] == "cot"]["correct"].mean()

    strong_nl4opt = _clean_nl4opt(
        _load(RESULTS_DIR / "nl4opt_gpt5_pot_strong.csv")
    )["correct"].mean()

    # CNMO (LiveMathBench) – CoT vs PoT Strong
    cot = _load(RESULTS_DIR / "livemathbench_all_gpt5_cot.csv")
    cot["split_norm"] = cot["split"].str.replace("_en", "", regex=False)
    cot_cnmo = cot[cot["split_norm"] == "CNMO"]["correct"].mean()

    pot_cnmo = _load(
        RESULTS_DIR / "livemathbench_cnmo_gpt5_pot_strong_full.csv"
    )["correct"].mean()

    rows = [
        {"Dataset": "NL4OPT (Optimization)", "Method": "CoT", "Accuracy": cot_nl4opt},
        {"Dataset": "NL4OPT (Optimization)", "Method": "PoT Strong", "Accuracy": strong_nl4opt},
        {"Dataset": "CNMO (Math Olympiad)", "Method": "CoT", "Accuracy": cot_cnmo},
        {"Dataset": "CNMO (Math Olympiad)", "Method": "PoT Strong", "Accuracy": pot_cnmo},
    ]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x="Dataset", y="Accuracy", hue="Method", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_title('The "PoT Gap": NL4OPT vs CNMO')
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3)

    output_path = FIGURES_DIR / "pot_gap_cross_dataset.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------
# 8) Hybrid router performance curve (task-mix vs accuracy)
# ---------------------------------------------------------------------

def plot_router_performance_curve() -> None:
    # Re-use the same accuracies as in the PoT-gap function
    strat = _load(RESULTS_DIR / "nl4opt_gpt5_strategies.csv")
    acc_opt_cot = strat[strat["strategy"] == "cot"]["correct"].mean()
    acc_opt_pot = _clean_nl4opt(
        _load(RESULTS_DIR / "nl4opt_gpt5_pot_strong.csv")
    )["correct"].mean()

    cot = _load(RESULTS_DIR / "livemathbench_all_gpt5_cot.csv")
    cot["split_norm"] = cot["split"].str.replace("_en", "", regex=False)
    acc_math_cot = cot[cot["split_norm"] == "CNMO"]["correct"].mean()
    acc_math_pot = _load(
        RESULTS_DIR / "livemathbench_cnmo_gpt5_pot_strong_full.csv"
    )["correct"].mean()

    ratios = np.linspace(0, 1, 101)  # 0 = all math, 1 = all optimization

    # Pure strategies
    pure_pot = ratios * acc_opt_pot + (1 - ratios) * acc_math_pot
    pure_cot = ratios * acc_opt_cot + (1 - ratios) * acc_math_cot

    # Hybrid router: assume router is 95% accurate at picking the right method
    router_acc = 0.95
    hybrid = (
        ratios * (router_acc * acc_opt_pot + (1 - router_acc) * acc_opt_cot)
        + (1 - ratios) * (router_acc * acc_math_cot + (1 - router_acc) * acc_math_pot)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ratios, pure_pot, linestyle="--", label="Pure PoT strategy", color=COLORS['secondary'], linewidth=2.5)
    ax.plot(ratios, pure_cot, linestyle="--", label="Pure CoT strategy", color=COLORS['primary'], linewidth=2.5)
    ax.plot(ratios, hybrid, label="Hybrid Router strategy", color=COLORS['tertiary'], linewidth=3)

    ax.set_xlabel("Dataset composition (1.0 = all optimization tasks)")
    ax.set_ylabel("Overall accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Router Performance vs Task Mix")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_path = FIGURES_DIR / "router_performance_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("✔ Plotting NL4OPT accuracy vs magnitude...")
    plot_nl4opt_accuracy_vs_magnitude()

    print("✔ Plotting NL4OPT accuracy–latency summary...")
    plot_nl4opt_accuracy_latency_summary()

    print("✔ Plotting NL4OPT Strong PoT diagnostics...")
    plot_nl4opt_strong_pot_diagnostics()

    print("✔ Plotting LiveMathBench per-split accuracy...")
    plot_lmb_per_split_accuracy()

    print("✔ Plotting LiveMathBench leaderboard...")
    plot_lmb_leaderboard()

    print("✔ Adding CoT vs CoT+SC heatmap...")
    plot_lmb_cot_sc_heatmap()

    print("✔ Plotting PoT gap cross-dataset chart...")
    plot_pot_gap_cross_dataset()

    print("✔ Plotting router performance curve...")
    plot_router_performance_curve()

    print(f"\n✅ All figures saved as individual PNG files in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
