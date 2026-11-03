import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import json
import os
import numpy as np

import envs.rewards as copula
import envs.regimes as regimes
from paths import DATA


def load_fitted_parameters(json_path=None):
    if json_path is None:
        json_path = os.path.join(DATA, "nj_params.json")
    with open(json_path) as f:
        return json.load(f)


def apply_nj_calibration(json_path=None):
    params = load_fitted_parameters(json_path)

    # json keys come back as strings
    copula.REGIME_SEVERITY = {int(k): v for k, v in params["regime_severity"].items()}

    # scale the existing pandemic loading, don't rebuild the scenario
    scale = params["pandemic_loading_scale"]
    original_loading = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    copula.SCENARIOS["pandemic"]["shared_loading"] = original_loading * scale

    print("Applied NJ-fitted calibration:")
    print(f"  REGIME_SEVERITY = {copula.REGIME_SEVERITY}")
    print(f"  pandemic loading = {copula.SCENARIOS['pandemic']['shared_loading']} (scale={scale:.4f})")
    print(f"  spillover_strength = {params['spillover_strength']:.4f}")
    return params


import envs.actions as env


def make_nj_fitted_control_env(json_path=None, scenario="pandemic", condition="gaussian"):
    params = apply_nj_calibration(json_path)
    return env.make_control_env(spillover_strength=params["spillover_strength"],
                                 scenario=scenario, condition=condition), params


if __name__ == "__main__":
    print("apply nj calibration and see if the mean check still holds")
    hospital_env, params = make_nj_fitted_control_env()

    import envs.surge as combined
    combined_env = combined.make_env(spillover_strength=params["spillover_strength"],
                                      scenario="pandemic", condition="gaussian")
    M = combined.closed_form_mean(combined_env)
    for st_idx in [0, 5, 10, 15]:
        mc_mean = combined.monte_carlo_mean(combined_env, st_idx, n_samples=20000, horizon=200, seed=st_idx)
        max_diff = np.abs(M[st_idx] - mc_mean).max()
        print(f"  state {st_idx:2d}  max|closed-form - MC| = {max_diff:.4f}")

    rng = np.random.default_rng(0)
    n = 300_000
    samples = copula.sample_shortfall(2, 2, "pandemic", "gaussian", rng, n)
    total_H = copula.region_total_shortfall(samples, "H")
    total_S = copula.region_total_shortfall(samples, "S")
    sim_corr = np.corrcoef(total_H, total_S)[0, 1]
    target = params.get("source_calibration", {}).get("cross_region_correlation_inpatient", "see data/nj_calib.json")
    print(f"  simulated cross-region correlation at Peak/Peak: {sim_corr:.4f}  (target: {target})")
