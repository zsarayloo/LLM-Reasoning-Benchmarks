import numpy as np
from utils.reasoning_vector import build_mean_diff_vector

H_pos = np.load("H_pos.npy")
H_neg = np.load("H_neg.npy")

c = build_mean_diff_vector(H_pos, H_neg)
np.save("reasoning_vector.npy", c)
