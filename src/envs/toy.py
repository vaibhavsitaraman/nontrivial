import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import numpy as np
import itertools

N_STATES = 2
N_ACTIONS = 2
STATE_NAMES = ["Low", "High"]
ACTION_NAMES = ["Conservative", "Aggressive"]

GAMMA = 0.9
# transitions ignore the action on purpose, so only the reward risk changes
TRANSITION_MATRIX = np.array([[0.7, 0.3], [0.4, 0.6]])

REWARD_PARAMS = {
    (0, 0): (1.0, 0.3),
    (0, 1): (1.2, 2.8),
    (1, 0): (0.5, 0.3),
    (1, 1): (1.4, 2.2),
}

CVAR_ALPHA = 0.1

assert np.allclose(TRANSITION_MATRIX.sum(axis=1), 1.0)


def sample_transition(state_idx, action_idx, rng):
    batch = state_idx.shape[0]
    reward = np.zeros(batch)
    next_state_idx = np.empty(batch, dtype=np.int64)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            mask = (state_idx == s) & (action_idx == a)
            n = int(mask.sum())
            if n > 0:
                mu, sigma = REWARD_PARAMS[(s, a)]
                reward[mask] = rng.normal(mu, sigma, size=n)
                next_state_idx[mask] = rng.choice(N_STATES, size=n, p=TRANSITION_MATRIX[s])
    return reward, next_state_idx


def evaluate_deterministic_policy(policy, n_mc=100_000, horizon=200, seed=0, alpha=CVAR_ALPHA):
    # freeze a policy, then mean/var are just an mrp again
    mu_arr = np.array([REWARD_PARAMS[(s, policy[s])][0] for s in range(N_STATES)])
    sig_arr = np.array([REWARD_PARAMS[(s, policy[s])][1] for s in range(N_STATES)])

    I = np.eye(N_STATES)
    P = TRANSITION_MATRIX
    M = np.linalg.solve(I - GAMMA * P, mu_arr)
    var_of_M = (P @ (M ** 2)) - (P @ M) ** 2
    c = sig_arr ** 2 + (GAMMA ** 2) * var_of_M
    V = np.linalg.solve(I - (GAMMA ** 2) * P, c)

    rng = np.random.default_rng(seed)
    cvars = []
    for s0 in range(N_STATES):
        state = np.full(n_mc, s0, dtype=np.int64)
        ret = np.zeros(n_mc)
        discount = 1.0
        for t in range(horizon):
            m = mu_arr[state]
            sd = sig_arr[state]
            r = rng.normal(m, sd)
            ret += discount * r
            discount *= GAMMA
            next_state = np.empty(n_mc, dtype=np.int64)
            for s in range(N_STATES):
                mask = state == s
                n = int(mask.sum())
                if n > 0:
                    next_state[mask] = rng.choice(N_STATES, size=n, p=P[s])
            state = next_state
        q = np.quantile(ret, alpha)
        cvars.append(ret[ret <= q].mean())
    return M, V, np.array(cvars)


def brute_force_optimal_policy(alpha=CVAR_ALPHA, seed=0):
    # only 4 policies, just try them all
    best_policy, best_score, all_results = None, -np.inf, []
    for actions in itertools.product(range(N_ACTIONS), repeat=N_STATES):
        policy = {s: a for s, a in enumerate(actions)}
        M, V, cvars = evaluate_deterministic_policy(policy, seed=seed, alpha=alpha)
        score = cvars.sum()
        all_results.append((policy, M, V, cvars, score))
        if score > best_score:
            best_score, best_policy = score, policy
    return best_policy, all_results


if __name__ == "__main__":
    print("Brute-force evaluation of all 4 deterministic policies:")
    best_policy, all_results = brute_force_optimal_policy()
    for policy, M, V, cvars, score in all_results:
        pname = [ACTION_NAMES[policy[s]] for s in range(N_STATES)]
        marker = "  <-- CVaR-optimal" if policy == best_policy else ""
        print(f"  policy={pname}  mean={np.round(M,3)}  var={np.round(V,3)}  "
              f"CVaR_{CVAR_ALPHA}={np.round(cvars,3)}{marker}")

    mean_optimal = max(all_results, key=lambda r: r[1].sum())[0]
    print(f"\nMean-optimal policy:  {[ACTION_NAMES[mean_optimal[s]] for s in range(N_STATES)]}")
    print(f"CVaR-optimal policy:  {[ACTION_NAMES[best_policy[s]] for s in range(N_STATES)]}")
    differ = mean_optimal != best_policy
    print(f"policies differ: {differ}")
