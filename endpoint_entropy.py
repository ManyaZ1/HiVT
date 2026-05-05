from scipy.stats import gaussian_kde
import numpy as np

def mode_entropy(endpoints):  # endpoints: (K, 2)
    kde = gaussian_kde(endpoints.T)
    # evaluate on a grid and compute entropy
    vals = kde(endpoints.T)
    vals = vals / vals.sum()
    return -np.sum(vals * np.log(vals + 1e-8))