import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import os
import glob
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
from nets.joint import compute_greedy_actions_multivariate
from nets.indep import compute_greedy_actions_independent
from nets.floq import train_floq, compute_greedy_actions_floq
from nets.particles_ctrl import train_sliced_cramer_control, compute_greedy_actions_sliced_cramer
from nets.flow import ConditionalVelocityNetSA
from paths import CKPT, PLOTS, DATA

SPILLOVER_STRENGTH = 0.7
SCENARIO = "pandemic"
CONDITION = "gaussian"

N_STEPS = 15000
BATCH_SIZE = 512
N_ODE_STEPS_TRAIN = 25
RECOMPUTE_GREEDY_EVERY = 25

SC_N_PARTICLES = 64
SC_N_DIRECTIONS = 8

TABULAR_N_PARTICLES = 10000
TABULAR_N_ITERATIONS = 100

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
        print("computing a fresh tabular reference")
        Z_ref, greedy_action_ref, cvar_table_ref, history_ref = tabular_greedy_cvar_value_iteration_hospital(
            hospital_env, n_particles=TABULAR_N_PARTICLES, n_iterations=TABULAR_N_ITERATIONS, seed=SEED)
        last_policies = history_ref[-15:]
        stable_per_state = [all(p[s] == last_policies[0][s] for p in last_policies) for s in range(env.N_STATES)]
        with open(REF_PATH, "wb") as f:
            pickle.dump(dict(Z_ref=Z_ref, greedy_action_ref=greedy_action_ref, stable_per_state=stable_per_state), f)

    print(f"training floq, {N_STEPS} steps")
    t0 = time.time()
    floq_model, floq_target, floq_losses, _ = train_floq(
        hospital_env, dim=env.N_DIM, weights=SCALARIZATION_WEIGHTS,
        n_steps=N_STEPS, batch_size=BATCH_SIZE, n_ode_steps_train=N_ODE_STEPS_TRAIN,
        recompute_greedy_every=RECOMPUTE_GREEDY_EVERY, device=DEVICE, seed=SEED, log_every=1000)
    floq_time = time.time() - t0
    print(f"floq took {floq_time:.1f}s")
    torch.save(floq_model.state_dict(), os.path.join(CKPT, "floq.pt"))

    print(f"training sliced cramer, {N_STEPS} steps")
    t0 = time.time()
    sc_model, sc_target, sc_losses, _ = train_sliced_cramer_control(
        hospital_env, dim=env.N_DIM, weights=SCALARIZATION_WEIGHTS, alpha=CVAR_ALPHA,
        n_particles=SC_N_PARTICLES, n_directions=SC_N_DIRECTIONS,
        n_steps=N_STEPS, batch_size=BATCH_SIZE,
        recompute_greedy_every=RECOMPUTE_GREEDY_EVERY, device=DEVICE, seed=SEED, log_every=1000)
    sc_time = time.time() - t0
    print(f"sliced took {sc_time:.1f}s")
    torch.save(sc_model.state_dict(), os.path.join(CKPT, "sliced.pt"))

    with open(os.path.join(CKPT, "baseline_meta.pkl"), "wb") as f:
        pickle.dump(dict(floq_losses=floq_losses, sc_losses=sc_losses, floq_time=floq_time, sc_time=sc_time), f)

    plt.figure(figsize=(9, 5))
    for label, losses, color in [("floq", floq_losses, "purple"), ("sliced", sc_losses, "teal")]:
        window = max(1, len(losses) // 150)
        rolling = np.convolve(losses, np.ones(window) / window, mode="valid") if window > 1 else losses
        plt.plot(range(len(rolling)), rolling, label=label, linewidth=1.5, color=color)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(PLOTS, "baseline_loss.png")
    plt.savefig(loss_path, dpi=150)

    floq_greedy_action, floq_value_table = compute_greedy_actions_floq(
        floq_target, env.N_STATES, env.N_ACTIONS, env.N_DIM, SCALARIZATION_WEIGHTS,
        device=DEVICE, n_ode_steps=EVAL_ODE_STEPS)
    sc_greedy_action, sc_cvar_table = compute_greedy_actions_sliced_cramer(
        sc_target, env.N_STATES, env.N_ACTIONS, SCALARIZATION_WEIGHTS, CVAR_ALPHA, device=DEVICE)

    joint_greedy_action, joint_cvar_table = None, None
    if os.path.exists(JOINT_PATH):
        print("loading joint model")
        joint_model = ConditionalVelocityNetSA(dim=env.N_DIM, n_states=env.N_STATES, n_actions=env.N_ACTIONS)
        joint_model.load_state_dict(torch.load(JOINT_PATH, map_location=DEVICE))
        joint_model.eval()
        joint_greedy_action, joint_cvar_table = compute_greedy_actions_multivariate(
            joint_model, env.N_STATES, env.N_ACTIONS, env.N_DIM, SCALARIZATION_WEIGHTS, CVAR_ALPHA,
            n_cvar_samples=8000, n_ode_steps=EVAL_ODE_STEPS, device=DEVICE)

    indep_greedy_action, indep_cvar_table = None, None
    indep_paths = sorted(glob.glob(os.path.join(CKPT, "indep_*.pt")))
    if len(indep_paths) == env.N_DIM:
        print("loading independent models")
        indep_models = []
        for d in range(env.N_DIM):
            m = ConditionalVelocityNetSA(dim=1, n_states=env.N_STATES, n_actions=env.N_ACTIONS)
            m.load_state_dict(torch.load(os.path.join(CKPT, f"indep_{d}.pt"), map_location=DEVICE))
            m.eval()
            indep_models.append(m)
        indep_greedy_action, indep_cvar_table = compute_greedy_actions_independent(
            indep_models, env.N_STATES, env.N_ACTIONS, SCALARIZATION_WEIGHTS, CVAR_ALPHA,
            n_cvar_samples=8000, n_ode_steps=EVAL_ODE_STEPS, device=DEVICE)
    else:
        print(f"indep checkpoints {len(indep_paths)}/{env.N_DIM}, skipping")

    methods = [("tabular", None, None), ("floq", floq_greedy_action, floq_value_table),
               ("sliced", sc_greedy_action, sc_cvar_table)]
    if joint_greedy_action is not None:
        methods.append(("joint", joint_greedy_action, joint_cvar_table))
    if indep_greedy_action is not None:
        methods.append(("independent", indep_greedy_action, indep_cvar_table))

    results = []
    match_counts = {name: 0 for name, _, _ in methods if name != "tabular"}
    match_counts_stable = {name: 0 for name, _, _ in methods if name != "tabular"}
    n_stable = sum(stable_per_state)

    for st in range(env.N_STATES):
        h, s = st // regimes.N_REGIMES, st % regimes.N_REGIMES
        tabular_action = greedy_action_ref[st]
        tabular_cvar = cvar_of_particles(Z_ref[(st, tabular_action)], CVAR_ALPHA)
        row = dict(state=st, h=h, s=s, stable=stable_per_state[st], tabular_cvar=tabular_cvar)
        for name, greedy_action, value_table in methods:
            if name == "tabular":
                continue
            action = greedy_action[st]
            value = value_table[st, action]
            match = action == tabular_action
            row[f"{name}_value"] = value
            row[f"{name}_match"] = match
            match_counts[name] += int(match)
            if stable_per_state[st]:
                match_counts_stable[name] += int(match)
        results.append(row)

    for name in match_counts:
        print(f"  {name:15s}  {match_counts[name]}/{env.N_STATES}   {match_counts_stable[name]}/{n_stable}")

    method_names = [name for name, _, _ in methods]
    n_methods = len(method_names)
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(env.N_STATES)
    width = 0.8 / n_methods
    colors = {"tabular": "steelblue", "floq": "purple", "sliced": "teal",
              "joint": "darkorange", "independent": "firebrick"}
    for i, name in enumerate(method_names):
        vals = [r["tabular_cvar"] if name == "tabular" else r[f"{name}_value"] for r in results]
        ax.bar(x + (i - n_methods / 2 + 0.5) * width, vals, width, label=name, color=colors.get(name))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{regimes.REGIMES[r['h']][:4]}/{regimes.REGIMES[r['s']][:4]}" for r in results],
                        rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(PLOTS, "baseline_compare.png")
    plt.savefig(plot_path, dpi=150)

    summary_path = os.path.join(DATA, "baseline_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"n_steps={N_STEPS}  floq={floq_time:.1f}s  sliced={sc_time:.1f}s\n")
        for name in match_counts:
            f.write(f"  {name:15s}  {match_counts[name]}/{env.N_STATES}   {match_counts_stable[name]}/{n_stable}\n")
        for r in results:
            line = f"state {r['state']:2d} tab={r['tabular_cvar']:7.3f}"
            for name, _, _ in methods:
                if name == "tabular":
                    continue
                line += f"  {name}={r[f'{name}_value']:7.3f}"
            f.write(line + "\n")
    print(f"saved {loss_path}, {plot_path}, {summary_path}")


if __name__ == "__main__":
    main()
