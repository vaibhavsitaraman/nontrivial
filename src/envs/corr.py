import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import numpy as np
from types import SimpleNamespace

import envs.chain2d as base

N_STATES = base.N_STATES
STATE_NAMES = base.STATE_NAMES
REWARD_DIM = base.REWARD_DIM
TRANSITION_MATRIX = base.TRANSITION_MATRIX
GAMMA = base.GAMMA
REWARD_MEANS = base.REWARD_MEANS
REWARD_STDS = base.REWARD_STDS


def build_reward_covs(rho):
    # same stds as the 2d chain, off-diagonal set by rho
    covs = np.zeros((N_STATES, REWARD_DIM, REWARD_DIM))
    for s in range(N_STATES):
        s1, s2 = REWARD_STDS[s]
        covs[s] = [[s1 ** 2, rho * s1 * s2],
                   [rho * s1 * s2, s2 ** 2]]
    return covs


def closed_form_moments(rho):
    reward_covs = build_reward_covs(rho)
    I = np.eye(N_STATES)
    P = TRANSITION_MATRIX
    d = REWARD_DIM

    # means don't care about rho
    A_mean = I - GAMMA * P
    M = np.linalg.solve(A_mean, REWARD_MEANS)

    C = np.zeros((N_STATES, d, d))
    for s in range(N_STATES):
        p = TRANSITION_MATRIX[s]
        mean_M = p @ M
        E_MMt = np.einsum("n,ni,nj->ij", p, M, M)
        C[s] = E_MMt - np.outer(mean_M, mean_M)

    A_var_inv = np.linalg.inv(I - (GAMMA ** 2) * P)
    V = np.zeros((N_STATES, d, d))
    for i in range(d):
        for j in range(i, d):
            D_ij = reward_covs[:, i, j] + (GAMMA ** 2) * C[:, i, j]
            V_ij = A_var_inv @ D_ij
            V[:, i, j] = V_ij
            V[:, j, i] = V_ij

    return M, V


def sample_transition_with_rho(state_idx, rng, rho):
    reward_covs = build_reward_covs(rho)
    batch = state_idx.shape[0]
    reward = np.zeros((batch, REWARD_DIM))
    next_state_idx = np.empty(batch, dtype=np.int64)
    for s in range(N_STATES):
        mask = state_idx == s
        n = int(mask.sum())
        if n > 0:
            reward[mask] = rng.multivariate_normal(REWARD_MEANS[s], reward_covs[s], size=n)
            next_state_idx[mask] = rng.choice(N_STATES, size=n, p=TRANSITION_MATRIX[s])
    return reward, next_state_idx


def monte_carlo_return_samples_with_rho(start_state, n_samples, rho, horizon=200, seed=0):
    rng = np.random.default_rng(seed)
    state_idx = np.full(n_samples, start_state, dtype=np.int64)
    returns = np.zeros((n_samples, REWARD_DIM))
    discount = 1.0
    for t in range(horizon):
        reward, next_state_idx = sample_transition_with_rho(state_idx, rng, rho)
        returns += discount * reward
        discount *= GAMMA
        state_idx = next_state_idx
    return returns


def make_env(rho):
    # full 2d reward, rho baked in
    def _sample_transition(state_idx, rng, _rho=rho):
        return sample_transition_with_rho(state_idx, rng, _rho)
    return SimpleNamespace(N_STATES=N_STATES, GAMMA=GAMMA, sample_transition=_sample_transition)


def make_marginal_env(rho, dim_index):
    # only one column, so a model trained here never sees the other dim
    def _sample_transition(state_idx, rng, _rho=rho, _dim=dim_index):
        reward2d, next_state_idx = sample_transition_with_rho(state_idx, rng, _rho)
        return reward2d[:, _dim], next_state_idx
    return SimpleNamespace(N_STATES=N_STATES, GAMMA=GAMMA, sample_transition=_sample_transition)


if __name__ == "__main__":
    print("Self-check 1: rho=0 should exactly reproduce chain2d's numbers.")
    M0, V0 = closed_form_moments(0.0)
    M_base, V_base = base.closed_form_moments()
    print(f"  mean match: {np.allclose(M0, M_base)}")
    print(f"  cov match:  {np.allclose(V0, V_base)}")

    print("\nSelf-check 2: closed-form vs. Monte Carlo at rho=0.85")
    rho_test = 0.85
    M, V = closed_form_moments(rho_test)
    for s in range(N_STATES):
        mc = monte_carlo_return_samples_with_rho(s, n_samples=150_000, rho=rho_test, horizon=200, seed=s)
        mc_cov = np.cov(mc.T)
        corr_cf = V[s][0, 1] / np.sqrt(V[s][0, 0] * V[s][1, 1])
        corr_mc = mc_cov[0, 1] / np.sqrt(mc_cov[0, 0] * mc_cov[1, 1])
        print(f"\n  state {STATE_NAMES[s]}: closed-form mean={M[s]}  MC mean={mc.mean(axis=0)}")
        print(f"    closed-form var=[{V[s][0,0]:.3f},{V[s][1,1]:.3f}]  "
              f"MC var=[{mc_cov[0,0]:.3f},{mc_cov[1,1]:.3f}]")
        print(f"    correlation: closed-form={corr_cf:.4f}  MC={corr_mc:.4f}")

    print("\nSelf-check 3: per-dimension variance should ignore rho")
    _, V_lo = closed_form_moments(0.0)
    _, V_hi = closed_form_moments(0.85)
    diag_match = np.allclose(V_lo[:, 0, 0], V_hi[:, 0, 0]) and np.allclose(V_lo[:, 1, 1], V_hi[:, 1, 1])
    print(f"  diagonal (variance) unaffected by rho: {diag_match}")
