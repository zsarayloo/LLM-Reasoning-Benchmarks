# data_loader_livemathbench.py

import os
import pandas as pd
from typing import Optional


def _load_single_file(path: str, split_name: str) -> pd.DataFrame:
    """
    Load a single *.jsonl file with LiveMathBench format.
    Expect fields: "question", "answer".
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"LiveMathBench file not found: {path}")

    df = pd.read_json(path, lines=True)
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError(
            f"File {path} does not have required columns "
            f"'question' and 'answer'. Found: {df.columns.tolist()}"
        )

    df["split"] = split_name
    return df[["question", "answer", "split"]]


def load_livemathbench(
    split: str = "all",
    local_dir: str = "data/LiveMathBench",
    n_examples: Optional[int] = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Load LiveMathBench splits from local JSONL files.

    Args:
        split: one of ["all", "AMC_en", "CCEE_en", "CNMO_en", "WLPMC_en", "hard_en"]
        local_dir: directory where files are stored
        n_examples: sample size (None for full)
        random_state: for deterministic sampling

    Returns:
        DataFrame with columns: question, answer, split
    """

    available_splits = {
        "AMC_en": "AMC_en.jsonl",
        "CCEE_en": "CCEE_en.jsonl",
        "CNMO_en": "CNMO_en.jsonl",
        "WLPMC_en": "WLPMC_en.jsonl",
        "hard_en": "hard_en.jsonl",
    }

    # If 'all', load every split
    if split == "all":
        dfs = []
        for sname, fname in available_splits.items():
            path = os.path.join(local_dir, fname)
            dfs.append(_load_single_file(path, sname))
        df = pd.concat(dfs, axis=0).reset_index(drop=True)

    else:
        # single split
        if split not in available_splits:
            raise ValueError(
                f"Unknown split '{split}'. Must be one of {list(available_splits.keys())} or 'all'."
            )
        path = os.path.join(local_dir, available_splits[split])
        df = _load_single_file(path, split)

    # Sampling
    if n_examples is not None:
        df = df.sample(n=n_examples, random_state=random_state).reset_index(drop=True)
        print(f"[load_livemathbench] Loaded {len(df)} samples from split='{split}'")
    else:
        print(f"[load_livemathbench] Loaded FULL set: {len(df)} examples from split='{split}'")

    return df
