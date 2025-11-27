
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# plot_livemathbench_heatmaps.py

# plot_livemathbench_accuracy.py


RESULTS = Path("results")

def load(name):
    return pd.read_csv(RESULTS / name)

sns.set_theme(style="whitegrid", context="talk")

df_cot = load("livemathbench_all_gpt5_cot.csv")
df_sc = load("livemathbench_all_gpt5_cot_selfconsistency.csv")
df_llama = load("livemathbench_all_llama_cot.csv")
df_mistral = load("livemathbench_all_mistral7b_cot_sample.csv")

def summarize(df, acc_col, lat_col, method):
    df = df.copy()
    df["split_norm"] = df["split"].str.replace("_en", "", regex=False)
    g = df.groupby("split_norm").agg(
        accuracy=(acc_col, "mean"),
        latency=(lat_col, "mean")
    ).reset_index()
    g["method"] = method
    return g

summary = pd.concat([
    summarize(df_cot, "correct", "latency_sec", "GPT-5 CoT"),
    summarize(df_sc, "sc_correct", "mean_latency_sec", "GPT-5 CoT + SC"),
    summarize(df_llama, "correct", "latency_sec", "LLaMA-3 CoT"),
    summarize(df_mistral, "correct", "latency_sec", "Mistral-7B CoT"),
])

plt.figure(figsize=(12, 6))
sns.barplot(
    data=summary,
    x="split_norm",
    y="accuracy",
    hue="method",
    palette="tab10"
)

plt.ylim(0,1)
plt.title("LiveMathBench – Accuracy per Split")
plt.xlabel("Competition Split")
plt.ylabel("Accuracy")
plt.tight_layout()

plt.savefig(RESULTS/"livemathbench_accuracy_clean.png", dpi=400)


# plot_nl4opt_summary.py



sns.set_theme(style="whitegrid", context="talk")
RESULTS = Path("results")

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
for name, file in files.items():
    df = pd.read_csv(RESULTS / file)
    rows.append({
        "method": name,
        "accuracy": df["correct"].mean(),
        "latency": df["latency_sec"].mean()
    })

summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(data=summary, y="method", x="accuracy", palette="Set2")
plt.title("NL4OPT – Objective Accuracy")
plt.savefig(RESULTS/"nl4opt_accuracy_clean.png", dpi=400)

# plot_pot_by_magnitude_clean.py


sns.set_theme(style="whitegrid", context="talk")
RESULTS = Path("results")

pot = pd.read_csv(RESULTS/"nl4opt_gpt5_pot_full.csv").rename(
    columns={"example_idx":"example", "correct":"correct_pot"})
strong = pd.read_csv(RESULTS/"nl4opt_gpt5_pot_strong.csv").rename(
    columns={"correct":"correct_strong"})

merged = pot.merge(strong, on="example")
merged["mag"] = merged["gt"].abs()

bins = [0,10,100,1000,10000,100000,1e9]
labels = ["0–10","10–100","100–1k","1k–10k","10k–100k","≥100k"]
merged["mag_bin"] = pd.cut(merged["mag"], bins=bins, labels=labels)

stats = merged.groupby("mag_bin").agg(
    pot_acc=("correct_pot","mean"),
    strong_acc=("correct_strong","mean")
).reset_index()

long = stats.melt(id_vars="mag_bin", value_vars=["pot_acc","strong_acc"],
                  var_name="method", value_name="accuracy")
long["method"] = long["method"].map({"pot_acc":"PoT baseline","strong_acc":"Strong PoT"})

plt.figure(figsize=(12,6))
sns.barplot(data=long, x="mag_bin", y="accuracy", hue="method", palette="Set2")
plt.title("NL4OPT – PoT vs Strong PoT by Objective Magnitude")
plt.tight_layout()
plt.savefig(RESULTS/"nl4opt_pot_vs_strong_clean.png", dpi=400)
