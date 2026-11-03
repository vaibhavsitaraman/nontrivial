import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

import envs.chain as mrp_env
from nets.flow import ode_sample
from nets.td import train_distributional_td
from paths import PLOTS

EVAL_SAMPLES = 5000
EVAL_ODE_STEPS = 50


def main():
    device = "cpu"
    M_true, V_true = mrp_env.closed_form_moments()
    for s in range(mrp_env.N_STATES):
        print(f"  state {mrp_env.STATE_NAMES[s]}: mean={M_true[s]:.4f}  var={V_true[s]:.4f}")

    print("training")
    model, losses = train_distributional_td(n_steps=8000, device=device)

    fig, axes = plt.subplots(1, mrp_env.N_STATES, figsize=(4.2 * mrp_env.N_STATES, 4))
    model_samples_per_state = {}
    w1_per_state = {}

    for s in range(mrp_env.N_STATES):
        state_idx = torch.full((EVAL_SAMPLES,), s, dtype=torch.long, device=device)
        model_samples = ode_sample(model, state_idx, EVAL_ODE_STEPS, device=device).squeeze(-1).numpy()
        mc_samples = mrp_env.monte_carlo_return_samples(s, n_samples=20000, horizon=200, seed=100 + s)

        learned_mean, learned_var = model_samples.mean(), model_samples.var()
        w1 = wasserstein_distance(model_samples, mc_samples)
        model_samples_per_state[s] = model_samples
        w1_per_state[s] = w1

        print(f"\n  state {mrp_env.STATE_NAMES[s]}:")
        print(f"    closed-form   mean={M_true[s]:.4f}  var={V_true[s]:.4f}")
        print(f"    learned model mean={learned_mean:.4f}  var={learned_var:.4f}   "
              f"(abs error: mean={abs(learned_mean - M_true[s]):.4f}, var={abs(learned_var - V_true[s]):.4f})")
        print(f"    Wasserstein-1: {w1:.4f}")

        ax = axes[s]
        ax.hist(mc_samples, bins=60, density=True, alpha=0.5, label="MC ground truth", color="tab:blue")
        ax.hist(model_samples, bins=60, density=True, alpha=0.5, label="learned", color="tab:orange")
        ax.set_title(f"State {mrp_env.STATE_NAMES[s]}  (W1={w1:.3f})")
        ax.set_xlabel("return")
        if s == 0:
            ax.set_ylabel("density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    dist_path = os.path.join(PLOTS, "td_dist.png")
    plt.savefig(dist_path, dpi=150)
    print(f"saved {dist_path}")

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("training step")
    plt.ylabel("flow-matching TD loss")
    plt.title("Training loss")
    plt.tight_layout()
    loss_path = os.path.join(PLOTS, "td_loss.png")
    plt.savefig(loss_path, dpi=150)
    print(f"saved {loss_path}")

    mean_thresh, var_thresh, w1_thresh = 0.3, 0.7, 0.5
    all_pass = True
    for s in range(mrp_env.N_STATES):
        learned_mean = model_samples_per_state[s].mean()
        learned_var = model_samples_per_state[s].var()
        mean_ok = abs(learned_mean - M_true[s]) < mean_thresh
        var_ok = abs(learned_var - V_true[s]) < var_thresh
        w1_ok = w1_per_state[s] < w1_thresh
        state_pass = mean_ok and var_ok and w1_ok
        all_pass &= state_pass
        print(f"  state {mrp_env.STATE_NAMES[s]}: {'PASS' if state_pass else 'FAIL'}  "
              f"(mean_ok={mean_ok}, var_ok={var_ok}, w1_ok={w1_ok})")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
