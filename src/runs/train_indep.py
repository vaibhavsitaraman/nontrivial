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
from nets.indep import train_independent_baseline, compute_greedy_actions_independent
from nets.joint import compute_greedy_actions_multivariate
from nets.flow import ConditionalVelocityNetSA
from paths import CKPT, PLOTS, DATA

SPILLOVER_STRENGTH = 0.7
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

REF_PATH = os.path.join(CKPT, "reference.pkl")
JOINT_PATH = os.path.join(CKPT, "joint.pt")


def main():
    hospital_env = env.make_control_env(spillover_strength=SPILLOVER_STRENGTH, scenario=SCENARIO, condition=CONDITION)

    if os.path.exists(REF_PATH):
        print(f"loading {REF_PATH}")
        with open(REF_PATH, "rb") as f:
            ref = pickle.load(f)
        Z_ref, greedy_action_ref, stable_per_state = ref["Z_ref"], ref["greedy_action_ref"], ref["stable_per_state"]
    else:
        print("no saved reference, computing one")
        t0 = time.time()
        Z_ref, greedy_action_ref, cvar_table_ref, history_ref = tabular_greedy_cvar_value_iteration_hospital(
            hospital_env, n_particles=TABULAR_N_PARTICLES, n_iterations=TABULAR_N_ITERATIONS, seed=SEED)
        print(f"tabular took {time.time() - t0:.1f}s")
        last_policies = history_ref[-15:]
        stable_per_state = [all(p[s] == last_policies[0][s] for p in last_policies) for s in range(env.N_STATES)]
        with open(REF_PATH, "wb") as f:
            pickle.dump(dict(Z_ref=Z_ref, greedy_action_ref=greedy_action_ref, stable_per_state=stable_per_state), f)

    print(f"training 6 separate nets, {N_STEPS} steps")
    t0 = time.time()
    models, target_models, losses, greedy_history = train_independent_baseline(
        hospital_env, dim=env.N_DIM, weights=SCALARIZATION_WEIGHTS, alpha=CVAR_ALPHA,
        n_steps=N_STEPS, batch_size=BATCH_SIZE,
        n_ode_steps_train=N_ODE_STEPS_TRAIN, n_cvar_samples=N_CVAR_SAMPLES_TRAIN,
        recompute_greedy_every=RECOMPUTE_GREEDY_EVERY, device=DEVICE, seed=SEED, log_every=500)
    train_time = time.time() - t0
    print(f"training took {train_time:.1f}s")

    for d, model_d in enumerate(models):
        torch.save(model_d.state_dict(), os.path.join(CKPT, f"indep_{d}.pt"))
    with open(os.path.join(CKPT, "indep_meta.pkl"), "wb") as f:
        pickle.dump(dict(losses=losses, greedy_history=greedy_history, train_time=train_time), f)

    plt.figure(figsize=(8, 5))
    for d in range(env.N_DIM):
        window = max(1, len(losses[d]) // 100)
        rolling = np.convolve(losses[d], np.ones(window) / window, mode="valid") if window > 1 else losses[d]
        plt.plot(range(len(rolling)), rolling, label=f"dim {d}", linewidth=1.2)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend(fontsize=8)
    plt.tight_layout()
    loss_path = os.path.join(PLOTS, "indep_loss.png")
    plt.savefig(loss_path, dpi=150)

    indep_greedy_action, indep_cvar_table = compute_greedy_actions_independent(
        target_models, env.N_STATES, env.N_ACTIONS, SCALARIZATION_WEIGHTS, CVAR_ALPHA,
        n_cvar_samples=EVAL_SAMPLES, n_ode_steps=EVAL_ODE_STEPS, device=DEVICE)

    joint_model = None
    joint_greedy_action, joint_cvar_table = None, None
    if os.path.exists(JOINT_PATH):
        print("found joint checkpoint, adding it to the plot")
        joint_model = ConditionalVelocityNetSA(dim=env.N_DIM, n_states=env.N_STATES, n_actions=env.N_ACTIONS)
        joint_model.load_state_dict(torch.load(JOINT_PATH, map_location=DEVICE))
        joint_model.eval()
        joint_greedy_action, joint_cvar_table = compute_greedy_actions_multivariate(
            joint_model, env.N_STATES, env.N_ACTIONS, env.N_DIM, SCALARIZATION_WEIGHTS, CVAR_ALPHA,
            n_cvar_samples=EVAL_SAMPLES, n_ode_steps=EVAL_ODE_STEPS, device=DEVICE)

    results = []
    n_matches_indep = 0
    n_matches_joint = 0
    for st in range(env.N_STATES):
        h, s = st // regimes.N_REGIMES, st % regimes.N_REGIMES
        tabular_action = greedy_action_ref[st]
        indep_action = indep_greedy_action[st]
        tabular_cvar = cvar_of_particles(Z_ref[(st, tabular_action)], CVAR_ALPHA)
        indep_cvar = indep_cvar_table[st, indep_action]
        indep_match = indep_action == tabular_action
        n_matches_indep += int(indep_match)
        row = dict(state=st, h=h, s=s, stable=stable_per_state[st],
                   tabular_cvar=tabular_cvar, indep_cvar=indep_cvar, indep_match=indep_match)
        if joint_model is not None:
            joint_action = joint_greedy_action[st]
            joint_cvar = joint_cvar_table[st, joint_action]
            joint_match = joint_action == tabular_action
            n_matches_joint += int(joint_match)
            row.update(joint_cvar=joint_cvar, joint_match=joint_match)
        results.append(row)

    n_stable = sum(stable_per_state)
    n_matches_indep_stable = sum(r["indep_match"] for r in results if r["stable"])
    print(f"indep {n_matches_indep}/{env.N_STATES} overall, {n_matches_indep_stable}/{n_stable} stable")
    if joint_model is not None:
        n_matches_joint_stable = sum(r["joint_match"] for r in results if r["stable"])
        print(f"joint {n_matches_joint}/{env.N_STATES} overall, {n_matches_joint_stable}/{n_stable} stable")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(env.N_STATES)
    width = 0.25 if joint_model is not None else 0.35
    offset = -width if joint_model is not None else -width / 2
    ax.bar(x + offset, [r["tabular_cvar"] for r in results], width, label="tabular", color="steelblue")
    offset2 = 0 if joint_model is not None else width / 2
    ax.bar(x + offset2, [r["indep_cvar"] for r in results], width, label="independent", color="firebrick")
    if joint_model is not None:
        ax.bar(x + width, [r["joint_cvar"] for r in results], width, label="joint", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{regimes.REGIMES[r['h']][:4]}/{regimes.REGIMES[r['s']][:4]}" for r in results],
                        rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(PLOTS, "indep_compare.png")
    plt.savefig(plot_path, dpi=150)

    summary_path = os.path.join(DATA, "indep_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"n_steps={N_STEPS}  train_time={train_time:.1f}s\n")
        f.write(f"indep {n_matches_indep}/{env.N_STATES}, stable {n_matches_indep_stable}/{n_stable}\n")
        if joint_model is not None:
            f.write(f"joint {n_matches_joint}/{env.N_STATES}, stable {n_matches_joint_stable}/{n_stable}\n")
        for r in results:
            line = f"state {r['state']:2d} tab={r['tabular_cvar']:7.3f} indep={r['indep_cvar']:7.3f}"
            if joint_model is not None:
                line += f" joint={r['joint_cvar']:7.3f}"
            f.write(line + "\n")
    print(f"saved {loss_path}, {plot_path}, {summary_path}")


if __name__ == "__main__":
    main()
