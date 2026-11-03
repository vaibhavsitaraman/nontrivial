import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import numpy as np
from types import SimpleNamespace

import envs.regimes as regimes
import envs.rewards as copula

N_REGIMES = regimes.N_REGIMES
N_STATES = N_REGIMES * N_REGIMES
GAMMA = 0.9


def state_to_regimes(state_idx):
    h = state_idx // N_REGIMES
    s = state_idx % N_REGIMES
    return h, s


def make_env(spillover_strength, scenario, condition):
    Q = regimes.build_joint_transition_matrix(spillover_strength)

    def sample_transition(state_idx, rng):
        batch = state_idx.shape[0]
        reward = np.zeros((batch, copula.N_DIM))
        next_state_idx = np.empty(batch, dtype=np.int64)

        for st in range(N_STATES):
            mask = state_idx == st
            n = int(mask.sum())
            if n == 0:
                continue
            h, s = state_to_regimes(st)
            reward[mask] = copula.sample_shortfall(h, s, scenario, condition, rng, n)
            next_state_idx[mask] = rng.choice(N_STATES, size=n, p=Q[st])

        return reward, next_state_idx

    return SimpleNamespace(N_STATES=N_STATES, GAMMA=GAMMA, sample_transition=sample_transition,
                            Q=Q, spillover_strength=spillover_strength, scenario=scenario, condition=condition)


def closed_form_mean(env):
    # shock mean is zero, so the return mean is just the regime means
    reward_means = np.zeros((N_STATES, copula.N_DIM))
    for st in range(N_STATES):
        h, s = state_to_regimes(st)
        reward_means[st] = copula.mean_vector(h, s)

    I = np.eye(N_STATES)
    M = np.linalg.solve(I - GAMMA * env.Q, reward_means)
    return M


def monte_carlo_mean(env, start_state, n_samples=20000, horizon=200, seed=0):
    rng = np.random.default_rng(seed)
    state_idx = np.full(n_samples, start_state, dtype=np.int64)
    returns = np.zeros((n_samples, copula.N_DIM))
    discount = 1.0
    for t in range(horizon):
        reward, next_state_idx = env.sample_transition(state_idx, rng)
        returns += discount * reward
        discount *= GAMMA
        state_idx = next_state_idx
    return returns.mean(axis=0)


def simulate_trajectory_joint_overload_rate(env, n_trajectories, horizon, threshold_H, threshold_S, seed=0):
    rng = np.random.default_rng(seed)
    state_idx = rng.integers(0, N_STATES, size=n_trajectories)
    overload_count = 0
    total_count = 0
    for t in range(horizon):
        reward = np.zeros((n_trajectories, copula.N_DIM))
        for st in np.unique(state_idx):
            mask = state_idx == st
            n = int(mask.sum())
            h, s = state_to_regimes(st)
            reward[mask] = copula.sample_shortfall(h, s, env.scenario, env.condition, rng, n)
        total_H = copula.region_total_shortfall(reward, "H")
        total_S = copula.region_total_shortfall(reward, "S")
        overload_count += int(np.sum((total_H > threshold_H) & (total_S > threshold_S)))
        total_count += n_trajectories

        next_state_idx = np.empty(n_trajectories, dtype=np.int64)
        for st in np.unique(state_idx):
            mask = state_idx == st
            n = int(mask.sum())
            next_state_idx[mask] = rng.choice(N_STATES, size=n, p=env.Q[st])
        state_idx = next_state_idx

    return overload_count / total_count


if __name__ == "__main__":
    print("Self-check 1: closed-form mean vs monte carlo.")
    env = make_env(spillover_strength=0.7, scenario="pandemic", condition="gaussian")
    M = closed_form_mean(env)
    for st in [0, 5, 10, 15]:
        h, s = state_to_regimes(st)
        mc_mean = monte_carlo_mean(env, st, n_samples=20000, horizon=200, seed=st)
        max_diff = np.abs(M[st] - mc_mean).max()
        print(f"  state {st:2d} (H={regimes.REGIMES[h]:8s} S={regimes.REGIMES[s]:8s})  "
              f"max|closed-form - MC| = {max_diff:.4f}")

    print("\nSelf-check 2: mean shouldn't care about gaussian vs student-t.")
    env_t = make_env(spillover_strength=0.7, scenario="pandemic", condition="student_t")
    M_t = closed_form_mean(env_t)
    print(f"  closed-form mean identical across condition: {np.allclose(M, M_t)}")

    print("\nSelf-check 3: full-trajectory joint overload vs spillover")
    for spill in [0.0, 0.4, 0.8]:
        env_g = make_env(spillover_strength=spill, scenario="pandemic", condition="gaussian")
        threshold_H, threshold_S = copula.calibrate_overload_thresholds(2, 2, "pandemic", percentile=0.97)
        rate = simulate_trajectory_joint_overload_rate(env_g, n_trajectories=4000, horizon=150,
                                                          threshold_H=threshold_H, threshold_S=threshold_S, seed=1)
        print(f"  spillover={spill:.1f}   full-trajectory joint-overload rate = {rate:.5f}")
