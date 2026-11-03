import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import os
import sys
import pickle
import torch
import numpy as np

import envs.corr as env
from nets.flow import ode_sample
from nets.td import train_distributional_td
from nets.metrics import energy_distance, weighted_cvar
from paths import CKPT

EVAL_SAMPLES = 5000
EVAL_ODE_STEPS = 50
MC_SAMPLES = 15000
CVAR_ALPHA = 0.1
CVAR_WEIGHTS = [0.5, 0.5]
N_STEPS = 5000


def sample_independent_combined(model_dim0, model_dim1, state_idx, device="cpu"):
    s0 = ode_sample(model_dim0, state_idx, EVAL_ODE_STEPS, device=device).numpy()
    s1 = ode_sample(model_dim1, state_idx, EVAL_ODE_STEPS, device=device).numpy()
    return np.concatenate([s0, s1], axis=1)


def run_one_rho(rho, device="cpu"):
    rng = np.random.default_rng(7)
    print(f"rho = {rho}  (n_steps={N_STEPS})")
    M_true, V_true = env.closed_form_moments(rho)

    print("  joint")
    joint_model, joint_losses = train_distributional_td(
        env=env.make_env(rho), dim=2, n_steps=N_STEPS, device=device, seed=0)
    print("  indep 0")
    model_dim0, _ = train_distributional_td(
        env=env.make_marginal_env(rho, 0), dim=1, n_steps=N_STEPS, device=device, seed=1)
    print("  indep 1")
    model_dim1, _ = train_distributional_td(
        env=env.make_marginal_env(rho, 1), dim=1, n_steps=N_STEPS, device=device, seed=2)

    state_results = []
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
        state_results.append(dict(
            rho=rho, state=s, true_corr=true_corr, joint_corr=joint_corr, indep_corr=indep_corr,
            cvar_true=cvar_true, cvar_joint=cvar_joint, cvar_indep=cvar_indep,
            edist_joint=edist_joint, edist_indep=edist_indep,
            joint_samples=joint_samples, indep_samples=indep_samples, mc_samples=mc_samples,
        ))

    out_path = os.path.join(CKPT, f"rho_{rho}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(dict(rho=rho, state_results=state_results, joint_losses=joint_losses), f)
    print(f"saved {out_path}")


if __name__ == "__main__":
    rho = float(sys.argv[1])
    run_one_rho(rho)
