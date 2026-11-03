import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import json
import os
import sys
import numpy as np
from scipy.optimize import minimize_scalar

import envs.regimes as regimes
import envs.rewards as copula
from paths import DATA


def load_nj_calibration(json_path=None):
    if json_path is None:
        json_path = os.path.join(DATA, "nj_calib.json")
    with open(json_path) as f:
        return json.load(f)


def fit_regime_severity(calib, region_key_H="inpatient_covid_util_pct_H", region_key_S="inpatient_covid_util_pct_S"):
    # one shared severity scale, averaged across the two regions
    h_stats = calib[region_key_H]
    s_stats = calib[region_key_S]

    baseline = min(h_stats["min"], s_stats["min"])
    quartiles_h, quartiles_s = h_stats["quartiles"], s_stats["quartiles"]
    q1_avg = (quartiles_h[0] + quartiles_s[0]) / 2
    median_avg = (quartiles_h[1] + quartiles_s[1]) / 2
    peak_anchor = max(h_stats["max"], s_stats["max"])

    # pin peak at 3 so it matches the old synthetic scale
    scale = 3.0 / (peak_anchor - baseline)
    severity = {
        0: 0.0,
        1: round((q1_avg - baseline) * scale, 3),
        2: 3.0,
        3: round((median_avg - baseline) * scale, 3),
    }
    return severity, baseline, peak_anchor


def fit_pandemic_loading_scale(calib, target_correlation_key="cross_region_correlation_inpatient"):
    target_corr = calib.get(target_correlation_key)
    if target_corr is None:
        print(f"  WARNING: {target_correlation_key} missing, leaving scale at 1.0")
        return 1.0

    base_loading = copula.SCENARIOS["pandemic"]["shared_loading"]
    idio_std = copula.IDIO_STD
    ped_shock_std = copula.SCENARIOS["pandemic"]["independent_ped_shock_std"]

    H_idx = [copula.dim_index("H", b) for b in copula.BED_TYPES]
    S_idx = [copula.dim_index("S", b) for b in copula.BED_TYPES]

    def idio_var_sum(idx_list):
        total = sum(idio_std[i] ** 2 for i in idx_list)
        # pediatric shock is extra variance on top of the usual idio noise
        if copula.PED_ICU_H in idx_list or copula.PED_ICU_S in idx_list:
            total += ped_shock_std ** 2
        return total

    idio_H_sum = idio_var_sum(H_idx)
    idio_S_sum = idio_var_sum(S_idx)
    l_H_sum_base = sum(base_loading[i] for i in H_idx)
    l_S_sum_base = sum(base_loading[i] for i in S_idx)

    def implied_corr(scale):
        l_H = l_H_sum_base * scale
        l_S = l_S_sum_base * scale
        return (l_H * l_S) / np.sqrt((l_H ** 2 + idio_H_sum) * (l_S ** 2 + idio_S_sum))

    result = minimize_scalar(lambda scale: (implied_corr(scale) - target_corr) ** 2,
                             bounds=(0.0, 10.0), method="bounded")
    fitted_scale = result.x
    print(f"  target corr: {target_corr:.4f}")
    print(f"  fitted scale: {fitted_scale:.4f}  (implied corr {implied_corr(fitted_scale):.4f})")
    return fitted_scale


def fit_spillover_strength(calib, lag_key="1"):
    lagged = calib.get("lagged_cross_correlation_H_leads_S")
    if lagged is None or lag_key not in lagged:
        print("  lag-1 corr missing, sticking with spillover 0.7")
        return 0.7

    # json may have stored the lag as an int or a string
    if lag_key not in lagged and str(lag_key) in lagged:
        lag_key = str(lag_key)
    target_lag1_corr = lagged[lag_key] if lag_key in lagged else lagged[str(lag_key)]

    def loss(spill):
        return (regimes.exact_lag1_regime_correlation(spill) - target_lag1_corr) ** 2

    result = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded")
    fitted_spillover = result.x
    print(f"  target lag-1 corr: {target_lag1_corr:.4f}")
    print(f"  fitted spillover: {fitted_spillover:.4f}  "
          f"(model corr {regimes.exact_lag1_regime_correlation(fitted_spillover):.4f})")
    return fitted_spillover


def build_nj_fitted_calibration(json_path=None):
    calib = load_nj_calibration(json_path)

    print("fitting regime severity...")
    severity, baseline, peak = fit_regime_severity(calib)
    print(f"  {severity}  (baseline={baseline:.2f}%, peak={peak:.2f}%)")

    print("\nfitting pandemic loading scale...")
    loading_scale = fit_pandemic_loading_scale(calib)

    print("\nfitting spillover...")
    spillover = fit_spillover_strength(calib)

    return dict(
        regime_severity=severity,
        pandemic_loading_scale=loading_scale,
        spillover_strength=spillover,
        source_calibration=calib,
    )


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "nj_calib.json")
    print(f"loading {json_path}")
    result = build_nj_fitted_calibration(json_path)
    print(f"  severity: {result['regime_severity']}")
    print(f"  loading scale: {result['pandemic_loading_scale']:.4f}")
    print(f"  spillover: {result['spillover_strength']:.4f}")

    out = os.path.join(DATA, "nj_params.json")
    with open(out, "w") as f:
        json.dump({k: v for k, v in result.items() if k != "source_calibration"}, f, indent=2)
    print(f"saved {out}")
