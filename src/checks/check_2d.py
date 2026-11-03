import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

import envs.chain2d as env
from nets.flow import ode_sample
from nets.td import train_distributional_td
from nets.metrics import energy_distance
from paths import PLOTS

EVAL_SAMPLES = 5000
EVAL_ODE_STEPS = 50


def main():
    device = "cpu"
    rng = np.random.default_rng(42)

    M_true, V_true = env.closed_form_moments()
    for s in range(env.N_STATES):
        corr = V_true[s][0, 1] / np.sqrt(V_true[s][0, 0] * V_true[s][1, 1])
        print(f"  state {env.STATE_NAMES[s]}: mean={M_true[s]}  "
              f"var=[{V_true[s][0,0]:.3f}, {V_true[s][1,1]:.3f}]  induced corr={corr:.4f}")

    model, losses = train_distributional_td(env=env, dim=2, n_steps=8000, device=device)

    fig_scatter, axes_scatter = plt.subplots(1, env.N_STATES, figsize=(4.5 * env.N_STATES, 4.3))
    fig_marg, axes_marg = plt.subplots(2, env.N_STATES, figsize=(4.2 * env.N_STATES, 7))
    results = {}

    for s in range(env.N_STATES):
        state_idx = torch.full((EVAL_SAMPLES,), s, dtype=torch.long, device=device)
        model_samples = ode_sample(model, state_idx, EVAL_ODE_STEPS, device=device).numpy()
        mc_samples = env.monte_carlo_return_samples(s, n_samples=20000, horizon=200, seed=100 + s)

        learned_mean = model_samples.mean(axis=0)
        learned_cov = np.cov(model_samples.T)
        learned_corr = learned_cov[0, 1] / np.sqrt(learned_cov[0, 0] * learned_cov[1, 1])
        true_corr = V_true[s][0, 1] / np.sqrt(V_true[s][0, 0] * V_true[s][1, 1])

        w1_dim0 = wasserstein_distance(model_samples[:, 0], mc_samples[:, 0])
        w1_dim1 = wasserstein_distance(model_samples[:, 1], mc_samples[:, 1])
        edist = energy_distance(model_samples, mc_samples, rng)

        results[s] = dict(
            learned_mean=learned_mean, learned_cov=learned_cov, learned_corr=learned_corr,
            true_corr=true_corr, w1_dim0=w1_dim0, w1_dim1=w1_dim1, edist=edist,
        )

        print(f"\n  state {env.STATE_NAMES[s]}:")
        print(f"    closed-form mean={M_true[s]}   learned mean={learned_mean}")
        print(f"    corr true={true_corr:.4f} learned={learned_corr:.4f}")
        print(f"    W1 dim0={w1_dim0:.4f} dim1={w1_dim1:.4f}  energy={edist:.4f}")

        ax = axes_scatter[s]
        ax.scatter(mc_samples[:1500, 0], mc_samples[:1500, 1], s=4, alpha=0.3, label="MC", color="tab:blue")
        ax.scatter(model_samples[:1500, 0], model_samples[:1500, 1], s=4, alpha=0.3, label="learned", color="tab:orange")
        ax.set_title(f"State {env.STATE_NAMES[s]}\ncorr: true={true_corr:.3f} learned={learned_corr:.3f}")
        ax.set_xlabel("return dim 0")
        ax.set_ylabel("return dim 1")
        ax.legend(fontsize=7, markerscale=3)

        for k in range(2):
            ax_m = axes_marg[k, s]
            ax_m.hist(mc_samples[:, k], bins=50, density=True, alpha=0.5, color="tab:blue", label="MC")
            ax_m.hist(model_samples[:, k], bins=50, density=True, alpha=0.5, color="tab:orange", label="learned")
            ax_m.set_title(f"State {env.STATE_NAMES[s]}, dim {k}  (W1={[w1_dim0, w1_dim1][k]:.3f})")
            if s == 0:
                ax_m.set_ylabel("density")
            if k == 0:
                ax_m.legend(fontsize=7)

    fig_scatter.tight_layout()
    scatter_path = os.path.join(PLOTS, "td2d_scatter.png")
    fig_scatter.savefig(scatter_path, dpi=150)
    fig_marg.tight_layout()
    marg_path = os.path.join(PLOTS, "td2d_marginals.png")
    fig_marg.savefig(marg_path, dpi=150)

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.tight_layout()
    loss_path = os.path.join(PLOTS, "td2d_loss.png")
    plt.savefig(loss_path, dpi=150)
    print(f"saved {scatter_path}, {marg_path}, {loss_path}")

    mean_thresh, var_thresh, corr_thresh, w1_thresh, edist_thresh = 0.3, 0.8, 0.15, 0.6, 0.6
    all_pass = True
    for s in range(env.N_STATES):
        r = results[s]
        mean_ok = np.all(np.abs(r["learned_mean"] - M_true[s]) < mean_thresh)
        var_ok = (abs(r["learned_cov"][0, 0] - V_true[s][0, 0]) < var_thresh and
                  abs(r["learned_cov"][1, 1] - V_true[s][1, 1]) < var_thresh)
        corr_ok = abs(r["learned_corr"] - r["true_corr"]) < corr_thresh
        w1_ok = r["w1_dim0"] < w1_thresh and r["w1_dim1"] < w1_thresh
        edist_ok = r["edist"] < edist_thresh
        state_pass = mean_ok and var_ok and corr_ok and w1_ok and edist_ok
        all_pass &= state_pass
        print(f"  state {env.STATE_NAMES[s]}: {'PASS' if state_pass else 'FAIL'}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
