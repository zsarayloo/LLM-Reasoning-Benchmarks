import os
import numpy as np

# ---------------- paths ----------------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

H_POS_PATH = os.path.join(PROJECT_ROOT, "H_pos.npy")
H_NEG_PATH = os.path.join(PROJECT_ROOT, "H_neg.npy")
V_PATH     = os.path.join(PROJECT_ROOT, "model", "reasoning_vector.npy")


def main() -> None:
    if not os.path.exists(H_POS_PATH) or not os.path.exists(H_NEG_PATH):
        raise FileNotFoundError("H_pos.npy / H_neg.npy not found. Run collect_reasoning_data.py first.")
    if not os.path.exists(V_PATH):
        raise FileNotFoundError("reasoning_vector.npy not found. Run build_reasoning_vector.py first.")

    H_pos = np.load(H_POS_PATH)   # [n_pos, d]
    H_neg = np.load(H_NEG_PATH)   # [n_neg, d]
    v     = np.load(V_PATH)       # [d]

    print(f"H_pos shape: {H_pos.shape}")
    print(f"H_neg shape: {H_neg.shape}")
    print(f"v shape    : {v.shape}")

    # projections along the reasoning direction
    proj_pos = H_pos @ v          # [n_pos]
    proj_neg = H_neg @ v          # [n_neg]

    print("\n=== Projection stats along reasoning vector ===")
    print(f"pos mean = {proj_pos.mean():.4f}, std = {proj_pos.std():.4f}")
    print(f"neg mean = {proj_neg.mean():.4f}, std = {proj_neg.std():.4f}")

    print("\npos projections:", proj_pos)
    print("neg projections:", proj_neg)


if __name__ == "__main__":
    main()
