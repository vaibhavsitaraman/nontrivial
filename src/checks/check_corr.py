import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import envs.corr as env
from nets.flow import ode_sample
from nets.td import train_distributional_td
from nets.metrics import energy_distance, weighted_cvar
from paths import PLOTS

RHO_VALUES = [0.0, 0.5, 0.99]
EVAL_SAMPLES = 5000
EVAL_ODE_STEPS = 50
MC_SAMPLES = 20000
CVAR_ALPHA = 0.1
CVAR_WEIGHTS = [0.5, 0.5]


def train_joint_and_independent(rho, device="cpu"):
    print("    joint model")
    joint_model, _ = train_distributional_td(env=env.make_env(rho), dim=2, n_steps=8000, device=device, seed=0)
    print("    indep dim 0")
    model_dim0, _ = train_distributional_td(env=env.make_marginal_env(rho, 0), dim=1, n_steps=8000, device=device, seed=1)
    print("    indep dim 1")
    model_dim1, _ = train_distributional_td(env=env.make_marginal_env(rho, 1), dim=1, n_steps=8000, device=device, seed=2)
    return joint_model, model_dim0, model_dim1


def sample_independent_combined(model_dim0, model_dim1, state_idx, device="cpu"):
    # stack two separate draws, so any correlation is an accident
    s0 = ode_sample(model_dim0, state_idx, EVAL_ODE_STEPS, device=device).numpy()
    s1 = ode_sample(model_dim1, state_idx, EVAL_ODE_STEPS, device=device).numpy()
    return np.concatenate([s0, s1], axis=1)


def main():
    device = "cpu"
    rng = np.random.default_rng(7)
    all_results = []

    for rho in RHO_VALUES:
        print(f"rho = {rho}")
        M_true, V_true = env.closed_form_moments(rho)
        joint_model, model_dim0, model_dim1 = train_joint_and_independent(rho, device=device)

        for s in range(env.N_STATES):
            state_idx = torch.full((EVAL_SAMPLES,), s, dtype=torch.long, device=device)
            joint_samples = ode_sample(joint_model, state_idx, EVAL_ODE_STEPS, device=device).numpy()
            indep_samples = sample_independent_combined(model_dim0, model_dim1, state_idx, device=device)
            mc_samples = env.monte_carlo_return_samples_with_rho(s, n_samples=MC_SAMPLES, rho=rho, horizon=200, seed=1000 + s)

            true_corr = V_true[s][0, 1] / np.sqrt(V_true[s][0, 0] * V_true[s][1, 1])
            joint_cov = np.cov(joint_samples.T)
            joint_corr = joint_cov[0, 1] / np.sqrt(joint_cov[0, 0] * joint_cov[1, 1])
            indep_cov = np.cov(indep_samples.T)
            indep_corr = indep_cov[0, 1] / np.sqrt(indep_cov[0, 0] * indep_cov[1, 1])

            cvar_true = weighted_cvar(mc_samples, CVAR_WEIGHTS, alpha=CVAR_ALPHA)
            cvar_joint = weighted_cvar(joint_samples, CVAR_WEIGHTS, alpha=CVAR_ALPHA)
            cvar_indep = weighted_cvar(indep_samples, CVAR_WEIGHTS, alpha=CVAR_ALPHA)
            edist_joint = energy_distance(joint_samples, mc_samples, rng)
            edist_indep = energy_distance(indep_samples, mc_samples, rng)

            print(f"  state {env.STATE_NAMES[s]} true={true_corr:.3f} joint={joint_corr:.3f} indep={indep_corr:.3f}")
            all_results.append(dict(
                rho=rho, state=s, true_corr=true_corr, joint_corr=joint_corr, indep_corr=indep_corr,
                cvar_true=cvar_true, cvar_joint=cvar_joint, cvar_indep=cvar_indep,
                edist_joint=edist_joint, edist_indep=edist_indep,
                joint_samples=joint_samples, indep_samples=indep_samples, mc_samples=mc_samples,
            ))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    for label, key, style in [("ground truth", "true_corr", "o-"),
                               ("joint model", "joint_corr", "s-"),
                               ("independent baseline", "indep_corr", "^-")]:
        ys = [np.mean([r[key] for r in all_results if r["rho"] == rho]) for rho in RHO_VALUES]
        ax.plot(RHO_VALUES, ys, style, label=label)
    ax.set_xlabel("rho")
    ax.set_ylabel("return correlation")
    ax.set_title("does the model learn the correlation?")
    ax.legend(fontsize=8)

    ax = axes[1]
    for label, jkey, tkey, style in [("joint model", "cvar_joint", "cvar_true", "s-"),
                                      ("independent baseline", "cvar_indep", "cvar_true", "^-")]:
        ys = [np.mean([abs(r[jkey] - r[tkey]) for r in all_results if r["rho"] == rho]) for rho in RHO_VALUES]
        ax.plot(RHO_VALUES, ys, style, label=label)
    ax.set_xlabel("rho")
    ax.set_ylabel(f"|CVaR_{CVAR_ALPHA} error|")
    ax.set_title("cvar of 0.5*z0 + 0.5*z1")
    ax.legend(fontsize=8)

    ax = axes[2]
    for label, key, style in [("joint model", "edist_joint", "s-"),
                               ("independent baseline", "edist_indep", "^-")]:
        ys = [np.mean([r[key] for r in all_results if r["rho"] == rho]) for rho in RHO_VALUES]
        ax.plot(RHO_VALUES, ys, style, label=label)
    ax.set_xlabel("rho")
    ax.set_ylabel("energy distance")
    ax.set_title("joint fit")
    ax.legend(fontsize=8)

    plt.tight_layout()
    sweep_path = os.path.join(PLOTS, "rho_sweep.png")
    plt.savefig(sweep_path, dpi=150)

    rho_show = RHO_VALUES[-1]
    show_results = [r for r in all_results if r["rho"] == rho_show]
    fig2, axes2 = plt.subplots(1, env.N_STATES, figsize=(4.5 * env.N_STATES, 4.3))
    for r in show_results:
        s = r["state"]
        ax = axes2[s]
        ax.scatter(r["mc_samples"][:1200, 0], r["mc_samples"][:1200, 1], s=4, alpha=0.3, label="MC", color="tab:blue")
        ax.scatter(r["joint_samples"][:1200, 0], r["joint_samples"][:1200, 1], s=4, alpha=0.3, label="joint", color="tab:orange")
        ax.scatter(r["indep_samples"][:1200, 0], r["indep_samples"][:1200, 1], s=4, alpha=0.3, label="indep", color="tab:green")
        ax.set_title(f"State {env.STATE_NAMES[s]}, rho={rho_show}")
        ax.legend(fontsize=7, markerscale=3)
    plt.tight_layout()
    scatter_path = os.path.join(PLOTS, "rho_scatter.png")
    fig2.savefig(scatter_path, dpi=150)
    print(f"saved {sweep_path} and {scatter_path}")

    corr_thresh = 0.15
    all_pass = True
    for rho in RHO_VALUES:
        rho_results = [r for r in all_results if r["rho"] == rho]
        joint_corr_err = np.mean([abs(r["joint_corr"] - r["true_corr"]) for r in rho_results])
        indep_corr_err = np.mean([abs(r["indep_corr"] - r["true_corr"]) for r in rho_results])
        joint_ok = joint_corr_err < corr_thresh
        print(f"  rho={rho:.2f} joint_err={joint_corr_err:.4f} indep_err={indep_corr_err:.4f} "
              f"{'PASS' if joint_ok else 'FAIL'}")
        all_pass &= joint_ok
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
