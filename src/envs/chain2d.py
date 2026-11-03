import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import numpy as np

N_STATES = 3
STATE_NAMES = ["A", "B", "C"]
REWARD_DIM = 2

TRANSITION_MATRIX = np.array([
    [0.5, 0.3, 0.2],
    [0.2, 0.5, 0.3],
    [0.3, 0.2, 0.5],
])
GAMMA = 0.9

# dim 1 was picked so the shared trajectory doesn't invent a huge correlation
REWARD_MEANS = np.array([
    [1.0, 1.3],
    [-0.5, 0.9],
    [2.0, 0.85],
])

REWARD_STDS = np.array([
    [0.5, 0.8],
    [1.0, 0.4],
    [0.3, 0.6],
])
# diagonal covs: shocks are independent given the state
REWARD_COVS = np.stack([np.diag(REWARD_STDS[s] ** 2) for s in range(N_STATES)])

assert np.allclose(TRANSITION_MATRIX.sum(axis=1), 1.0), "P must be row-stochastic"


def closed_form_moments():
    I = np.eye(N_STATES)
    P = TRANSITION_MATRIX
    d = REWARD_DIM

    A_mean = I - GAMMA * P
    M = np.linalg.solve(A_mean, REWARD_MEANS)

    # how much the mean vector jitters just from the next state being random
    C = np.zeros((N_STATES, d, d))
    for s in range(N_STATES):
        p = TRANSITION_MATRIX[s]
        mean_M = p @ M
        E_MMt = np.einsum("n,ni,nj->ij", p, M, M)
        C[s] = E_MMt - np.outer(mean_M, mean_M)

    A_var = I - (GAMMA ** 2) * P
    A_var_inv = np.linalg.inv(A_var)

    # each cov entry is its own little linear system
    V = np.zeros((N_STATES, d, d))
    for i in range(d):
        for j in range(i, d):
            D_ij = REWARD_COVS[:, i, j] + (GAMMA ** 2) * C[:, i, j]
            V_ij = A_var_inv @ D_ij
            V[:, i, j] = V_ij
            V[:, j, i] = V_ij

    return M, V


def sample_transition(state_idx, rng):
    batch = state_idx.shape[0]
    reward = np.zeros((batch, REWARD_DIM))
    next_state_idx = np.empty(batch, dtype=np.int64)

    for s in range(N_STATES):
        mask = state_idx == s
        n = int(mask.sum())
        if n > 0:
            reward[mask] = rng.multivariate_normal(REWARD_MEANS[s], REWARD_COVS[s], size=n)
            next_state_idx[mask] = rng.choice(N_STATES, size=n, p=TRANSITION_MATRIX[s])

    return reward, next_state_idx


def monte_carlo_return_samples(start_state, n_samples, horizon=200, seed=0):
    rng = np.random.default_rng(seed)
    state_idx = np.full(n_samples, start_state, dtype=np.int64)
    returns = np.zeros((n_samples, REWARD_DIM))
    discount = 1.0

    for t in range(horizon):
        reward, next_state_idx = sample_transition(state_idx, rng)
        returns += discount * reward
        discount *= GAMMA
        state_idx = next_state_idx

    return returns


if __name__ == "__main__":
    M, V = closed_form_moments()
    print("Closed-form vs. Monte Carlo self-check:")
    for s in range(N_STATES):
        mc = monte_carlo_return_samples(s, n_samples=200_000, horizon=200, seed=s)
        mc_mean = mc.mean(axis=0)
        mc_cov = np.cov(mc.T)
        corr_cf = V[s][0, 1] / np.sqrt(V[s][0, 0] * V[s][1, 1])
        corr_mc = mc_cov[0, 1] / np.sqrt(mc_cov[0, 0] * mc_cov[1, 1])
        print(f"\n  state {STATE_NAMES[s]}:")
        print(f"    closed-form mean = {M[s]},  MC mean = {mc_mean}")
        print(f"    closed-form cov  =\n{V[s]}")
        print(f"    MC cov           =\n{mc_cov}")
        print(f"    induced correlation coeff: closed-form={corr_cf:.4f}  MC={corr_mc:.4f}")
