import os
import sys
import numpy as np

# ---------------------------------------------------------------------
# Resolve project root the same way as in collect_reasoning_data.py
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

H_POS_PATH = os.path.join(PROJECT_ROOT, "H_pos.npy")
H_NEG_PATH = os.path.join(PROJECT_ROOT, "H_neg.npy")

# Where to save the reasoning vector
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
OUT_PATH = os.path.join(MODEL_DIR, "reasoning_vector.npy")


def main() -> None:
    if not os.path.exists(H_POS_PATH) or not os.path.exists(H_NEG_PATH):
        raise FileNotFoundError(
            f"Could not find H_pos/H_neg.\n"
            f"Expected:\n  {H_POS_PATH}\n  {H_NEG_PATH}\n"
            f"Run collect_reasoning_data.py first."
        )

    H_pos = np.load(H_POS_PATH)   # shape: [n_pos, d]
    H_neg = np.load(H_NEG_PATH)   # shape: [n_neg, d]

    print(f"[build_reasoning_vector] H_pos shape: {H_pos.shape}")
    print(f"[build_reasoning_vector] H_neg shape: {H_neg.shape}")

    mean_pos = H_pos.mean(axis=0)
    mean_neg = H_neg.mean(axis=0)

    v = mean_pos - mean_neg  # core "reasoning direction"

    # Optional: L2-normalize (helps keep scale under control)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(OUT_PATH, v)

    print(f"[build_reasoning_vector] Saved reasoning vector to {OUT_PATH}")
    print(f"[build_reasoning_vector] Vector dim = {v.shape[0]}")


if __name__ == "__main__":
    main()
