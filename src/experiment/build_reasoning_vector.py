import numpy as np
import os
import sys

# Add src/ to sys.path so "utils" is importable
THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from utils.reasoning_vector import build_mean_diff_vector

H_pos = np.load("H_pos.npy")
H_neg = np.load("H_neg.npy")

c = build_mean_diff_vector(H_pos, H_neg)
np.save("reasoning_vector.npy", c)
