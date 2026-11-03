import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import numpy as np
from scipy.spatial.distance import cdist


def energy_distance(X, Y, rng, n_sub=1000):
    # subsample so the pairwise distances don't explode
    X = X[rng.choice(len(X), size=min(n_sub, len(X)), replace=False)]
    Y = Y[rng.choice(len(Y), size=min(n_sub, len(Y)), replace=False)]
    d_xy = cdist(X, Y).mean()
    d_xx = cdist(X, X).mean()
    d_yy = cdist(Y, Y).mean()
    return 2 * d_xy - d_xx - d_yy


def cvar(samples_1d, alpha=0.1):
    # mean of the worst alpha-fraction (low rewards are the bad ones)
    q = np.quantile(samples_1d, alpha)
    tail = samples_1d[samples_1d <= q]
    return tail.mean()


def weighted_cvar(samples_2d, weights, alpha=0.1):
    # squash the vector down with weights, then take the bad tail
    combined = samples_2d @ np.asarray(weights)
    return cvar(combined, alpha=alpha)
