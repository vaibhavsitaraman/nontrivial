import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import numpy as np
import envs.toy as env


def cvar_of_particles(particles, alpha):
    q = np.quantile(particles, alpha)
    tail = particles[particles <= q]
    return tail.mean()


def tabular_greedy_cvar_value_iteration(n_particles=20000, n_iterations=150,
                                          alpha=env.CVAR_ALPHA, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    # a cloud of particles for every state-action pair
    Z = {(s, a): rng.normal(0, 1, size=n_particles) for s in range(env.N_STATES) for a in range(env.N_ACTIONS)}

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
                reward, next_state_idx = env.sample_transition(state_idx, action_idx, rng)

                # pull successor particles from the greedy action
                z_next = np.empty(n_particles)
                for sp in range(env.N_STATES):
                    mask = next_state_idx == sp
                    n = int(mask.sum())
                    if n > 0:
                        a_star = greedy_action[sp]
                        z_next[mask] = rng.choice(Z[(sp, a_star)], size=n, replace=True)

                new_Z[(s, a)] = reward + env.GAMMA * z_next
        Z = new_Z

        history.append((dict(greedy_action), {s: list(cvar_table[s]) for s in cvar_table}))
        if verbose and (it % 25 == 0 or it == n_iterations - 1):
            gp = [env.ACTION_NAMES[greedy_action[s]] for s in range(env.N_STATES)]
            print(f"  iter {it:4d}  greedy policy={gp}  "
                  f"CVaR(s=0)={cvar_table[0]}  CVaR(s=1)={cvar_table[1]}")

    return Z, greedy_action, cvar_table, history


if __name__ == "__main__":
    print("Running exact tabular greedy-CVaR value iteration (no neural network)...")
    Z, greedy_action, cvar_table, history = tabular_greedy_cvar_value_iteration()

    print("\nConverged greedy policy:")
    for s in range(env.N_STATES):
        print(f"  state {env.STATE_NAMES[s]}: {env.ACTION_NAMES[greedy_action[s]]}   "
              f"CVaR per action={np.round(cvar_table[s], 3)}")

    last_policies = [h[0] for h in history[-20:]]
    stable = all(p == last_policies[0] for p in last_policies)
    print(f"\nStability check (last 20 iters): {'STABLE' if stable else 'NOT STABLE'}")

    print("\nDistribution moments of the converged Z[s, greedy_action[s]]:")
    for s in range(env.N_STATES):
        z = Z[(s, greedy_action[s])]
        print(f"  state {env.STATE_NAMES[s]}: mean={z.mean():.3f}  var={z.var():.3f}  "
              f"CVaR_{env.CVAR_ALPHA}={cvar_of_particles(z, env.CVAR_ALPHA):.3f}")
