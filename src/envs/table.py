import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import numpy as np
import envs.actions as env

CVAR_ALPHA = 0.1
# icu counts a bit more than a regular bed, both regions equal
SCALARIZATION_WEIGHTS = np.array([0.8, 1.2, 1.2, 0.8, 1.2, 1.2])
SCALARIZATION_WEIGHTS = SCALARIZATION_WEIGHTS / SCALARIZATION_WEIGHTS.sum()


def cvar_of_particles(particles_vec, alpha, weights=SCALARIZATION_WEIGHTS):
    scalarized = particles_vec @ weights
    q = np.quantile(scalarized, alpha)
    tail = scalarized[scalarized <= q]
    return tail.mean()


def tabular_greedy_cvar_value_iteration_hospital(hospital_env, n_particles=4000, n_iterations=60,
                                                    alpha=CVAR_ALPHA, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    Z = {(s, a): rng.normal(0, 1, size=(n_particles, env.N_DIM))
         for s in range(env.N_STATES) for a in range(env.N_ACTIONS)}

    history = []
    for it in range(n_iterations):
        greedy_action = {}
        cvar_table = {}
        for s in range(env.N_STATES):
            cvars = [cvar_of_particles(Z[(s, a)], alpha) for a in range(env.N_ACTIONS)]
            cvar_table[s] = cvars
            greedy_action[s] = int(np.argmax(cvars))

        new_Z = {}
        for s in range(env.N_STATES):
            for a in range(env.N_ACTIONS):
                state_idx = np.full(n_particles, s, dtype=np.int64)
                action_idx = np.full(n_particles, a, dtype=np.int64)
                reward, next_state_idx = hospital_env.sample_transition(state_idx, action_idx, rng)

                z_next = np.empty((n_particles, env.N_DIM))
                for sp in range(env.N_STATES):
                    mask = next_state_idx == sp
                    n = int(mask.sum())
                    if n > 0:
                        a_star = greedy_action[sp]
                        idx = rng.choice(n_particles, size=n, replace=True)
                        z_next[mask] = Z[(sp, a_star)][idx]

                new_Z[(s, a)] = reward + hospital_env.GAMMA * z_next
        Z = new_Z

        history.append(dict(greedy_action))
        if verbose and (it % 10 == 0 or it == n_iterations - 1):
            gp = [env.ACTION_NAMES[greedy_action[s]] for s in range(env.N_STATES)]
            print(f"  iter {it:3d}  greedy actions per state: {gp}")

    return Z, greedy_action, cvar_table, history


if __name__ == "__main__":
    print("tabular cvar on the hospital env (spillover=0.7, pandemic, gaussian)")
    hospital_env = env.make_control_env(spillover_strength=0.7, scenario="pandemic", condition="gaussian")
    Z, greedy_action, cvar_table, history = tabular_greedy_cvar_value_iteration_hospital(hospital_env)

    print("\nConverged greedy policy:")
    import envs.regimes as regimes
    for st in range(env.N_STATES):
        h, s = st // regimes.N_REGIMES, st % regimes.N_REGIMES
        print(f"  state {st:2d}  H={regimes.REGIMES[h]:8s} S={regimes.REGIMES[s]:8s}   "
              f"greedy action = {env.ACTION_NAMES[greedy_action[st]]}")

    last_policies = [h for h in history[-15:]]
    stable = all(p == last_policies[0] for p in last_policies)
    print(f"\nlast 15 iters stable: {stable}")
