import numpy as np
from sklearn.decomposition import PCA

def build_mean_diff_vector(H_pos, H_neg):
    μ_pos = np.mean(H_pos, axis=0)
    μ_neg = np.mean(H_neg, axis=0)
    return μ_pos - μ_neg

def build_pca_vector(H_pos, H_neg):
    diffs = np.vstack([H_pos - np.mean(H_pos, 0),
                       H_neg - np.mean(H_neg, 0)])
    pca = PCA(n_components=1)
    pca.fit(diffs)
    return pca.components_[0]    # shape: (d,)
