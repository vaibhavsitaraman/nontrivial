import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import os
import time
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

import envs.nj_env as nj_env
import envs.regimes as regimes
from envs.table import (
    tabular_greedy_cvar_value_iteration_hospital, cvar_of_particles,
    SCALARIZATION_WEIGHTS, CVAR_ALPHA,
)
from nets.joint import train_distributional_td_control_multivariate, compute_greedy_actions_multivariate
import envs.actions as env
from paths import CKPT, PLOTS, DATA

SCENARIO = "pandemic"
CONDITION = "gaussian"

N_STEPS = 15000
BATCH_SIZE = 512
N_CVAR_SAMPLES_TRAIN = 800
N_ODE_STEPS_TRAIN = 25
RECOMPUTE_GREEDY_EVERY = 25

TABULAR_N_PARTICLES = 10000
TABULAR_N_ITERATIONS = 100

EVAL_SAMPLES = 8000
EVAL_ODE_STEPS = 50

DEVICE = "cpu"
SEED = 0

SYNTH_REF = os.path.join(CKPT, "reference.pkl")
LIT_REF = os.path.join(CKPT, "lit_reference.pkl")


def main():
    hospital_env, fitted_params = nj_env.make_nj_fitted_control_env(scenario=SCENARIO, condition=CONDITION)

    print("tabular reference on the nj-fitted env")
    t0 = time.time()
    Z_ref, greedy_action_ref, cvar_table_ref, history_ref = tabular_greedy_cvar_value_iteration_hospital(
        hospital_env, n_particles=TABULAR_N_PARTICLES, n_iterations=TABULAR_N_ITERATIONS, seed=SEED)
    print(f"tabular took {time.time() - t0:.1f}s")

    last_policies = history_ref[-15:]
    stable_per_state = [all(p[s] == last_policies[0][s] for p in last_policies) for s in range(env.N_STATES)]
    print(f"{sum(stable_per_state)}/{env.N_STATES} stable")

    with open(os.path.join(CKPT, "nj_reference.pkl"), "wb") as f:
        pickle.dump(dict(Z_ref=Z_ref, greedy_action_ref=greedy_action_ref, stable_per_state=stable_per_state), f)

    print(f"training, spillover={fitted_params['spillover_strength']:.4f}")
    t0 = time.time()
    model, target_model, losses, greedy_history = train_distributional_td_control_multivariate(
        hospital_env, dim=env.N_DIM, weights=SCALARIZATION_WEIGHTS, alpha=CVAR_ALPHA,
        n_steps=N_STEPS, batch_size=BATCH_SIZE,
        n_ode_steps_train=N_ODE_STEPS_TRAIN, n_cvar_samples=N_CVAR_SAMPLES_TRAIN,
        recompute_greedy_every=RECOMPUTE_GREEDY_EVERY, device=DEVICE, seed=SEED, log_every=500)
    train_time = time.time() - t0
    print(f"training took {train_time:.1f}s")

    torch.save(model.state_dict(), os.path.join(CKPT, "nj.pt"))
    with open(os.path.join(CKPT, "nj_meta.pkl"), "wb") as f:
        pickle.dump(dict(losses=losses, greedy_history=greedy_history, train_time=train_time,
                          fitted_params=fitted_params), f)

    plt.figure(figsize=(7, 4.5))
    plt.plot(losses, linewidth=0.8)
    window = max(1, len(losses) // 100)
    if window > 1:
        rolling = np.convolve(losses, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(losses)), rolling, linewidth=2, label="rolling avg")
        plt.legend()
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.tight_layout()
    loss_path = os.path.join(PLOTS, "nj_loss.png")
    plt.savefig(loss_path, dpi=150)

    learned_greedy_action, learned_cvar_table = compute_greedy_actions_multivariate(
        model, env.N_STATES, env.N_ACTIONS, env.N_DIM, SCALARIZATION_WEIGHTS, CVAR_ALPHA,
        n_cvar_samples=EVAL_SAMPLES, n_ode_steps=EVAL_ODE_STEPS, device=DEVICE)

    results = []
    n_matches = 0
    for st in range(env.N_STATES):
        h, s = st // regimes.N_REGIMES, st % regimes.N_REGIMES
        tabular_action = greedy_action_ref[st]
        learned_action = learned_greedy_action[st]
        match = tabular_action == learned_action
        n_matches += int(match)
        tabular_cvar = cvar_of_particles(Z_ref[(st, tabular_action)], CVAR_ALPHA)
        learned_cvar = learned_cvar_table[st, learned_action]
        results.append(dict(state=st, h=h, s=s, stable=stable_per_state[st],
                             tabular_action=tabular_action, learned_action=learned_action,
                             match=match, tabular_cvar=tabular_cvar, learned_cvar=learned_cvar))

    n_stable = sum(stable_per_state)
    n_matches_stable = sum(r["match"] for r in results if r["stable"])
    print(f"match {n_matches}/{env.N_STATES}, stable {n_matches_stable}/{n_stable}")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(env.N_STATES)
    tabular_cvars = [r["tabular_cvar"] for r in results]
    learned_cvars = [r["learned_cvar"] for r in results]

    # older references, if they happen to be sitting in checkpoints
    prior_refs = []
    if os.path.exists(SYNTH_REF):
        with open(SYNTH_REF, "rb") as f:
            ref = pickle.load(f)
        vals = [cvar_of_particles(ref["Z_ref"][(st, ref["greedy_action_ref"][st])], CVAR_ALPHA) for st in range(env.N_STATES)]
        prior_refs.append(("synthetic", vals, "gray"))
    if os.path.exists(LIT_REF):
        with open(LIT_REF, "rb") as f:
            ref = pickle.load(f)
        vals = [cvar_of_particles(ref["Z_ref"][(st, ref["greedy_action_ref"][st])], CVAR_ALPHA) for st in range(env.N_STATES)]
        prior_refs.append(("literature", vals, "seagreen"))

    n_bars = 2 + len(prior_refs)
    width = 0.8 / n_bars
    ax.bar(x + (0 - n_bars / 2 + 0.5) * width, tabular_cvars, width, label="nj tabular", color="steelblue")
    ax.bar(x + (1 - n_bars / 2 + 0.5) * width, learned_cvars, width, label="nj learned", color="darkorange")
    for i, (label, vals, color) in enumerate(prior_refs):
        ax.bar(x + (2 + i - n_bars / 2 + 0.5) * width, vals, width, label=label, color=color, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{regimes.REGIMES[r['h']][:4]}/{regimes.REGIMES[r['s']][:4]}" for r in results],
                        rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(PLOTS, "nj_compare.png")
    plt.savefig(plot_path, dpi=150)

    summary_path = os.path.join(DATA, "nj_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"n_steps={N_STEPS}  train_time={train_time:.1f}s\n")
        f.write(f"fitted: spillover={fitted_params['spillover_strength']} scale={fitted_params['pandemic_loading_scale']}\n")
        f.write(f"match {n_matches}/{env.N_STATES}, stable {n_matches_stable}/{n_stable}\n")
        for r in results:
            f.write(f"state {r['state']:2d}  H={regimes.REGIMES[r['h']]:8s} S={regimes.REGIMES[r['s']]:8s}  "
                    f"{'MATCH' if r['match'] else 'mismatch'}  "
                    f"CVaR tab={r['tabular_cvar']:.3f} learned={r['learned_cvar']:.3f}\n")
    print(f"saved {loss_path}, {plot_path}, {summary_path}")


if __name__ == "__main__":
    main()
