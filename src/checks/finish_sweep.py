import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import envs.corr as env
from paths import CKPT, PLOTS

RHO_VALUES = [0.0, 0.5, 0.99]
CVAR_ALPHA = 0.1


def load_all():
    all_results = []
    for rho in RHO_VALUES:
        path = os.path.join(CKPT, f"rho_{rho}.pkl")
        with open(path, "rb") as f:
            d = pickle.load(f)
        all_results.extend(d["state_results"])
    return all_results


def main():
    all_results = load_all()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    ax = axes[0]
    for label, key, style in [("ground truth", "true_corr", "o-"),
                               ("joint model", "joint_corr", "s-"),
                               ("independent baseline", "indep_corr", "^-")]:
        ys = [np.mean([r[key] for r in all_results if r["rho"] == rho]) for rho in RHO_VALUES]
        ax.plot(RHO_VALUES, ys, style, label=label)
    ax.set_xlabel("rho")
    ax.set_ylabel("return correlation")
    ax.legend(fontsize=8)

    ax = axes[1]
    for label, jkey, tkey, style in [("joint model", "cvar_joint", "cvar_true", "s-"),
                                      ("independent baseline", "cvar_indep", "cvar_true", "^-")]:
        ys = [np.mean([abs(r[jkey] - r[tkey]) for r in all_results if r["rho"] == rho]) for rho in RHO_VALUES]
        ax.plot(RHO_VALUES, ys, style, label=label)
    ax.set_xlabel("rho")
    ax.set_ylabel(f"|CVaR_{CVAR_ALPHA} error|")
    ax.legend(fontsize=8)

    ax = axes[2]
    for label, key, style in [("joint model", "edist_joint", "s-"),
                               ("independent baseline", "edist_indep", "^-")]:
        ys = [np.mean([r[key] for r in all_results if r["rho"] == rho]) for rho in RHO_VALUES]
        ax.plot(RHO_VALUES, ys, style, label=label)
    ax.set_xlabel("rho")
    ax.set_ylabel("energy distance")
    ax.legend(fontsize=8)

    plt.tight_layout()
    sweep_path = os.path.join(PLOTS, "rho_sweep.png")
    plt.savefig(sweep_path, dpi=150)
    print(f"saved {sweep_path}")

    rho_show = RHO_VALUES[-1]
    show_results = [r for r in all_results if r["rho"] == rho_show]
    fig2, axes2 = plt.subplots(1, env.N_STATES, figsize=(4.5 * env.N_STATES, 4.3))
    for r in show_results:
        s = r["state"]
        ax = axes2[s]
        ax.scatter(r["mc_samples"][:1200, 0], r["mc_samples"][:1200, 1], s=4, alpha=0.3, label="MC", color="tab:blue")
        ax.scatter(r["joint_samples"][:1200, 0], r["joint_samples"][:1200, 1], s=4, alpha=0.3, label="joint", color="tab:orange")
        ax.scatter(r["indep_samples"][:1200, 0], r["indep_samples"][:1200, 1], s=4, alpha=0.3, label="indep", color="tab:green")
        ax.set_title(f"state {env.STATE_NAMES[s]} rho={rho_show}")
        ax.legend(fontsize=7, markerscale=3)
    plt.tight_layout()
    scatter_path = os.path.join(PLOTS, "rho_scatter.png")
    fig2.savefig(scatter_path, dpi=150)
    print(f"saved {scatter_path}")

    corr_thresh = 0.15
    all_pass = True
    for rho in RHO_VALUES:
        rho_results = [r for r in all_results if r["rho"] == rho]
        joint_corr_err = np.mean([abs(r["joint_corr"] - r["true_corr"]) for r in rho_results])
        indep_corr_err = np.mean([abs(r["indep_corr"] - r["true_corr"]) for r in rho_results])
        joint_cvar_err = np.mean([abs(r["cvar_joint"] - r["cvar_true"]) for r in rho_results])
        indep_cvar_err = np.mean([abs(r["cvar_indep"] - r["cvar_true"]) for r in rho_results])
        joint_ok = joint_corr_err < corr_thresh
        ratio = indep_cvar_err / joint_cvar_err if joint_cvar_err else float("inf")
        print(f"  rho={rho:.2f} corr joint={joint_corr_err:.4f} indep={indep_corr_err:.4f} "
              f"cvar ratio={ratio:.2f}x {'PASS' if joint_ok else 'FAIL'}")
        all_pass &= joint_ok
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
