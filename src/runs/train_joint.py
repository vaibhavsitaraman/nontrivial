import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import os
import time
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

import envs.actions as env
import envs.regimes as regimes
from envs.table import (
    tabular_greedy_cvar_value_iteration_hospital, cvar_of_particles,
    SCALARIZATION_WEIGHTS, CVAR_ALPHA,
)
from nets.joint import train_distributional_td_control_multivariate, compute_greedy_actions_multivariate
from paths import CKPT, PLOTS, DATA

SPILLOVER_STRENGTH = 0.7
SCENARIO = "pandemic"
CONDITION = "gaussian"

N_STEPS = 15000
BATCH_SIZE = 512
N_CVAR_SAMPLES_TRAIN = 800
N_ODE_STEPS_TRAIN = 12
RECOMPUTE_GREEDY_EVERY = 25

TABULAR_N_PARTICLES = 10000
TABULAR_N_ITERATIONS = 100

EVAL_SAMPLES = 8000
EVAL_ODE_STEPS = 50

DEVICE = "cpu"
SEED = 0


def main():
    hospital_env = env.make_control_env(spillover_strength=SPILLOVER_STRENGTH, scenario=SCENARIO, condition=CONDITION)

    print("tabular reference")
    t0 = time.time()
    Z_ref, greedy_action_ref, cvar_table_ref, history_ref = tabular_greedy_cvar_value_iteration_hospital(
        hospital_env, n_particles=TABULAR_N_PARTICLES, n_iterations=TABULAR_N_ITERATIONS, seed=SEED)
    print(f"tabular took {time.time() - t0:.1f}s")

    # a state counts as stable if the last 15 iters all picked the same action
    last_policies = history_ref[-15:]
    stable_per_state = [all(p[s] == last_policies[0][s] for p in last_policies) for s in range(env.N_STATES)]
    print(f"{sum(stable_per_state)}/{env.N_STATES} states stable")

    with open(os.path.join(CKPT, "reference.pkl"), "wb") as f:
        pickle.dump(dict(Z_ref=Z_ref, greedy_action_ref=greedy_action_ref, stable_per_state=stable_per_state), f)

    print(f"training joint model, {N_STEPS} steps")
    t0 = time.time()
    model, target_model, losses, greedy_history = train_distributional_td_control_multivariate(
        hospital_env, dim=env.N_DIM, weights=SCALARIZATION_WEIGHTS, alpha=CVAR_ALPHA,
        n_steps=N_STEPS, batch_size=BATCH_SIZE,
        n_ode_steps_train=N_ODE_STEPS_TRAIN, n_cvar_samples=N_CVAR_SAMPLES_TRAIN,
        recompute_greedy_every=RECOMPUTE_GREEDY_EVERY, device=DEVICE, seed=SEED, log_every=500)
    train_time = time.time() - t0
    print(f"training took {train_time:.1f}s")

    torch.save(model.state_dict(), os.path.join(CKPT, "joint.pt"))
    with open(os.path.join(CKPT, "joint_meta.pkl"), "wb") as f:
        pickle.dump(dict(losses=losses, greedy_history=greedy_history, train_time=train_time), f)

    plt.figure(figsize=(7, 4.5))
    plt.plot(losses, linewidth=0.8)
    window = max(1, len(losses) // 100)
    if window > 1:
        rolling = np.convolve(losses, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(losses)), rolling, linewidth=2, label=f"rolling avg (window={window})")
        plt.legend()
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title(f"joint training loss (n_steps={N_STEPS})")
    plt.tight_layout()
    loss_path = os.path.join(PLOTS, "joint_loss.png")
    plt.savefig(loss_path, dpi=150)
    print(f"saved {loss_path}")

    print("evaluating")
    t0 = time.time()
    learned_greedy_action, learned_cvar_table = compute_greedy_actions_multivariate(
        model, env.N_STATES, env.N_ACTIONS, env.N_DIM, SCALARIZATION_WEIGHTS, CVAR_ALPHA,
        n_cvar_samples=EVAL_SAMPLES, n_ode_steps=EVAL_ODE_STEPS, device=DEVICE)
    print(f"eval took {time.time() - t0:.1f}s")

    results = []
    n_matches = 0
    for st in range(env.N_STATES):
        h, s = st // regimes.N_REGIMES, st % regimes.N_REGIMES
        tabular_action = greedy_action_ref[st]
        learned_action = learned_greedy_action[st]
        match = tabular_action == learned_action
        tabular_cvar = cvar_of_particles(Z_ref[(st, tabular_action)], CVAR_ALPHA)
        learned_cvar = learned_cvar_table[st, learned_action]
        results.append(dict(state=st, h=h, s=s, tabular_action=tabular_action, learned_action=learned_action,
                             match=match, stable=stable_per_state[st], tabular_cvar=tabular_cvar, learned_cvar=learned_cvar))
        n_matches += int(match)

    n_matches_stable = sum(r["match"] for r in results if r["stable"])
    n_stable = sum(stable_per_state)
    print(f"match {n_matches}/{env.N_STATES}, stable-only {n_matches_stable}/{n_stable}")

    for r in results:
        print(f"  state {r['state']:2d}  H={regimes.REGIMES[r['h']]:8s} S={regimes.REGIMES[r['s']]:8s}  "
              f"tabular={env.ACTION_NAMES[r['tabular_action']]:20s}  learned={env.ACTION_NAMES[r['learned_action']]:20s}  "
              f"{'MATCH' if r['match'] else 'mismatch'}  CVaR tab={r['tabular_cvar']:.3f} learned={r['learned_cvar']:.3f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    tabular_cvars = [r["tabular_cvar"] for r in results]
    learned_cvars = [r["learned_cvar"] for r in results]
    x = np.arange(len(results))
    width = 0.35
    ax.bar(x - width / 2, tabular_cvars, width, label="tabular", color="steelblue")
    ax.bar(x + width / 2, learned_cvars, width, label="learned", color="darkorange")
    for i, r in enumerate(results):
        ax.annotate("ok" if r["match"] else "no", (i, max(tabular_cvars[i], learned_cvars[i]) + 0.3),
                    ha="center", color="green" if r["match"] else "red", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{regimes.REGIMES[r['h']][:4]}/{regimes.REGIMES[r['s']][:4]}" for r in results],
                        rotation=45, ha="right")
    ax.set_ylabel(f"CVaR_{CVAR_ALPHA}")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(PLOTS, "joint_policy.png")
    plt.savefig(plot_path, dpi=150)

    summary_path = os.path.join(DATA, "joint_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"n_steps={N_STEPS}  train_time={train_time:.1f}s\n")
        f.write(f"match {n_matches}/{env.N_STATES}, stable {n_matches_stable}/{n_stable}\n")
        for r in results:
            f.write(f"state {r['state']:2d}  H={regimes.REGIMES[r['h']]:8s} S={regimes.REGIMES[r['s']]:8s}  "
                    f"{'MATCH' if r['match'] else 'mismatch'}  "
                    f"CVaR tab={r['tabular_cvar']:.3f} learned={r['learned_cvar']:.3f}\n")
    print(f"saved {plot_path} and {summary_path}")


if __name__ == "__main__":
    main()
