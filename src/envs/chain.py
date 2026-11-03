import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import numpy as np

# tiny 3-state chain, no actions yet
N_STATES = 3
STATE_NAMES = ["A", "B", "C"]

TRANSITION_MATRIX = np.array([
    [0.5, 0.3, 0.2],
    [0.2, 0.5, 0.3],
    [0.3, 0.2, 0.5],
])

REWARD_MEANS = np.array([1.0, -0.5, 2.0])
REWARD_STDS = np.array([0.5, 1.0, 0.3])

GAMMA = 0.9

assert np.allclose(TRANSITION_MATRIX.sum(axis=1), 1.0), "P must be row-stochastic"


def closed_form_moments():
    # solve the mean and variance bellman equations exactly
    I = np.eye(N_STATES)
    P = TRANSITION_MATRIX
    mu = REWARD_MEANS
    sigma2 = REWARD_STDS ** 2

    M = np.linalg.solve(I - GAMMA * P, mu)

    var_of_M_given_s = (P @ (M ** 2)) - (P @ M) ** 2
    c = sigma2 + (GAMMA ** 2) * var_of_M_given_s
    V = np.linalg.solve(I - (GAMMA ** 2) * P, c)

    return M, V


def sample_transition(state_idx, rng):
    batch = state_idx.shape[0]
    reward = rng.normal(REWARD_MEANS[state_idx], REWARD_STDS[state_idx])

    # next state depends on which row of P we're in
    next_state_idx = np.empty(batch, dtype=np.int64)
    for s in range(N_STATES):
        mask = state_idx == s
        n = mask.sum()
        if n > 0:
            next_state_idx[mask] = rng.choice(N_STATES, size=n, p=TRANSITION_MATRIX[s])

    return reward, next_state_idx


def monte_carlo_return_samples(start_state, n_samples, horizon=200, seed=0):
    # roll out a bunch of full returns from one start state
    rng = np.random.default_rng(seed)
    state_idx = np.full(n_samples, start_state, dtype=np.int64)
    returns = np.zeros(n_samples)
    discount = 1.0

    for t in range(horizon):
        reward = rng.normal(REWARD_MEANS[state_idx], REWARD_STDS[state_idx])
        returns += discount * reward
        discount *= GAMMA

        next_state_idx = np.empty(n_samples, dtype=np.int64)
        for s in range(N_STATES):
            mask = state_idx == s
            n = mask.sum()
            if n > 0:
                next_state_idx[mask] = rng.choice(N_STATES, size=n, p=TRANSITION_MATRIX[s])
        state_idx = next_state_idx

    return returns


if __name__ == "__main__":
    M, V = closed_form_moments()
    print("Closed-form check (should match Monte Carlo estimates closely):")
    for s in range(N_STATES):
        mc = monte_carlo_return_samples(s, n_samples=200_000, horizon=200, seed=s)
        print(f"  state {STATE_NAMES[s]}: closed-form mean={M[s]:.4f} var={V[s]:.4f}  |  "
              f"MC mean={mc.mean():.4f} var={mc.var():.4f}")
