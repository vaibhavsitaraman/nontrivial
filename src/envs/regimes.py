import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import numpy as np

REGIMES = ["Baseline", "Rising", "Peak", "Waning"]
N_REGIMES = len(REGIMES)

# hub's own epidemic cycle
P_H = np.array([
    [0.75, 0.20, 0.02, 0.03],
    [0.10, 0.55, 0.30, 0.05],
    [0.02, 0.08, 0.70, 0.20],
    [0.35, 0.05, 0.05, 0.55],
])
assert np.allclose(P_H.sum(axis=1), 1.0)


def build_P_S_given_H(spillover_strength):
    # spoke is slower on its own, then gets pulled toward whatever the hub is doing
    P_S_baseline = np.array([
        [0.85, 0.12, 0.01, 0.02],
        [0.15, 0.65, 0.15, 0.05],
        [0.05, 0.10, 0.75, 0.10],
        [0.30, 0.05, 0.05, 0.60],
    ])
    assert np.allclose(P_S_baseline.sum(axis=1), 1.0)

    P_S_given_H = np.zeros((N_REGIMES, N_REGIMES, N_REGIMES))
    for h in range(N_REGIMES):
        pulled = np.full(N_REGIMES, 0.05)
        pulled[h] = 1.0 - 0.05 * (N_REGIMES - 1)
        pulled_matrix = np.tile(pulled, (N_REGIMES, 1))
        P_S_given_H[h] = (1 - spillover_strength) * P_S_baseline + spillover_strength * pulled_matrix
        assert np.allclose(P_S_given_H[h].sum(axis=1), 1.0)
    return P_S_given_H


def joint_state_index(h, s):
    return h * N_REGIMES + s


def build_joint_transition_matrix(spillover_strength):
    # tomorrow's spoke depends on today's hub, not tomorrow's hub
    P_S_given_H = build_P_S_given_H(spillover_strength)
    n = N_REGIMES * N_REGIMES
    Q = np.zeros((n, n))
    for h in range(N_REGIMES):
        for s in range(N_REGIMES):
            i = joint_state_index(h, s)
            for hp in range(N_REGIMES):
                for sp in range(N_REGIMES):
                    j = joint_state_index(hp, sp)
                    Q[i, j] = P_H[h, hp] * P_S_given_H[h][s, sp]
    assert np.allclose(Q.sum(axis=1), 1.0)
    return Q


def stationary_distribution(Q, n_power_iters=5000):
    n = Q.shape[0]
    pi = np.full(n, 1.0 / n)
    for _ in range(n_power_iters):
        pi = pi @ Q
    return pi


def exact_lag1_regime_correlation(spillover_strength):
    # treat regimes as 0..3 just so we can quote one correlation number
    Q = build_joint_transition_matrix(spillover_strength)
    pi = stationary_distribution(Q)

    h_vals = np.array([h for h in range(N_REGIMES) for s in range(N_REGIMES)])

    h_t_mean = (pi * h_vals).sum()
    h_t_var = (pi * (h_vals - h_t_mean) ** 2).sum()

    joint_h_t_s_tp1 = np.zeros((N_REGIMES, N_REGIMES))
    for h in range(N_REGIMES):
        for s in range(N_REGIMES):
            i = joint_state_index(h, s)
            p_i = pi[i]
            for sp in range(N_REGIMES):
                p_sp_given_i = sum(Q[i, joint_state_index(hp, sp)] for hp in range(N_REGIMES))
                joint_h_t_s_tp1[h, sp] += p_i * p_sp_given_i

    s_tp1_mean = joint_h_t_s_tp1.sum(axis=0) @ np.arange(N_REGIMES)
    s_tp1_var = (joint_h_t_s_tp1.sum(axis=0) * (np.arange(N_REGIMES) - s_tp1_mean) ** 2).sum()

    cov = 0.0
    for h in range(N_REGIMES):
        for sp in range(N_REGIMES):
            cov += joint_h_t_s_tp1[h, sp] * (h - h_t_mean) * (sp - s_tp1_mean)

    corr = cov / np.sqrt(h_t_var * s_tp1_var)
    return corr


def simulate(spillover_strength, n_steps, seed=0):
    rng = np.random.default_rng(seed)
    P_S_given_H = build_P_S_given_H(spillover_strength)
    h = rng.integers(0, N_REGIMES)
    s = rng.integers(0, N_REGIMES)
    h_traj, s_traj = [h], [s]
    for t in range(n_steps):
        s_next = rng.choice(N_REGIMES, p=P_S_given_H[h][s])
        h_next = rng.choice(N_REGIMES, p=P_H[h])
        h, s = h_next, s_next
        h_traj.append(h)
        s_traj.append(s)
    return np.array(h_traj), np.array(s_traj)


if __name__ == "__main__":
    print("Self-check 1: exact stationary distribution vs a long sim (spillover=0.5).")
    spill = 0.5
    Q = build_joint_transition_matrix(spill)
    pi_exact = stationary_distribution(Q)
    print(f"  exact stationary distribution sums to {pi_exact.sum():.6f}")

    h_traj, s_traj = simulate(spill, n_steps=500_000, seed=0)
    burn_in = 1000
    h_traj, s_traj = h_traj[burn_in:], s_traj[burn_in:]
    empirical_joint = np.zeros(N_REGIMES * N_REGIMES)
    for h, s in zip(h_traj, s_traj):
        empirical_joint[joint_state_index(h, s)] += 1
    empirical_joint /= empirical_joint.sum()
    max_abs_diff = np.abs(pi_exact - empirical_joint).max()
    print(f"  max abs diff between exact stationary dist and empirical: {max_abs_diff:.4f}")

    print("\nSelf-check 2: lag-1 correlation vs spillover")
    for spill in [0.0, 0.2, 0.5, 0.8, 1.0]:
        corr_exact = exact_lag1_regime_correlation(spill)
        print(f"  spillover={spill:.1f}   exact corr(h_t, s_t+1) = {corr_exact:.4f}")

    print("\nSelf-check 3: exact lag-1 corr vs a sim at spillover=0.7")
    spill = 0.7
    corr_exact = exact_lag1_regime_correlation(spill)
    h_traj, s_traj = simulate(spill, n_steps=500_000, seed=1)
    h_t = h_traj[burn_in:-1]
    s_tp1 = s_traj[burn_in + 1:]
    corr_empirical = np.corrcoef(h_t, s_tp1)[0, 1]
    print(f"  exact corr={corr_exact:.4f}   empirical corr={corr_empirical:.4f}   "
          f"diff={abs(corr_exact - corr_empirical):.4f}")
