import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import numpy as np

REGIONS = ["H", "S"]
BED_TYPES = ["inpatient", "adult_icu", "pediatric_icu"]
DIM_NAMES = [f"{r}_{b}" for r in REGIONS for b in BED_TYPES]
N_DIM = len(DIM_NAMES)


def dim_index(region, bed_type):
    return REGIONS.index(region) * len(BED_TYPES) + BED_TYPES.index(bed_type)


PED_ICU_H = dim_index("H", "pediatric_icu")
PED_ICU_S = dim_index("S", "pediatric_icu")

# how bad each regime is, on a shared scale
REGIME_SEVERITY = {0: 0.0, 1: 1.5, 2: 3.0, 3: 1.2}

BASE_MEAN = np.array([0.5, 0.3, 0.2, 0.5, 0.3, 0.2])
SEVERITY_SENSITIVITY = np.array([1.0, 1.3, 0.8, 0.9, 1.2, 0.7])
IDIO_STD = np.array([0.4, 0.5, 0.4, 0.4, 0.5, 0.4])

SCENARIOS = {
    "isolated_pediatric": {
        "shared_loading": np.zeros(N_DIM),
        "independent_ped_shock_std": 1.2,
    },
    "regional_flu": {
        "shared_loading": np.array([0.9, 1.1, 0.1, 0.8, 1.0, 0.1]),
        "independent_ped_shock_std": 0.3,
    },
    "pandemic": {
        "shared_loading": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        "independent_ped_shock_std": 0.3,
    },
}


def mean_vector(h_regime, s_regime):
    # first 3 dims follow the hub, last 3 follow the spoke
    severity = np.array([REGIME_SEVERITY[h_regime]] * 3 + [REGIME_SEVERITY[s_regime]] * 3)
    return BASE_MEAN + SEVERITY_SENSITIVITY * severity


def sample_shortfall(h_regime, s_regime, scenario, condition, rng, n, t_df=4):
    params = SCENARIOS[scenario]
    mean = mean_vector(h_regime, s_regime)
    loading = params["shared_loading"]

    # common shock is unit variance either way
    if condition == "gaussian":
        common_shock = rng.normal(0, 1, size=n)
    elif condition == "student_t":
        raw = rng.standard_t(t_df, size=n)
        common_shock = raw / np.sqrt(t_df / (t_df - 2))
    else:
        raise ValueError(condition)

    shared_component = np.outer(common_shock, loading)
    idio = rng.normal(0, IDIO_STD, size=(n, N_DIM))

    # extra kid-icu noise that is not shared across regions
    ped_std = params["independent_ped_shock_std"]
    ped_shock_H = rng.normal(0, ped_std, size=n)
    ped_shock_S = rng.normal(0, ped_std, size=n)
    independent_ped = np.zeros((n, N_DIM))
    independent_ped[:, PED_ICU_H] = ped_shock_H
    independent_ped[:, PED_ICU_S] = ped_shock_S

    shortfall = mean[None, :] + shared_component + idio + independent_ped
    return shortfall


def region_total_shortfall(shortfall, region):
    idx = [dim_index(region, b) for b in BED_TYPES]
    return shortfall[:, idx].sum(axis=1)


def joint_overload_probability(shortfall, threshold_H, threshold_S):
    total_H = region_total_shortfall(shortfall, "H")
    total_S = region_total_shortfall(shortfall, "S")
    return float(np.mean((total_H > threshold_H) & (total_S > threshold_S)))


def cvar(samples_1d, alpha=0.1):
    q = np.quantile(samples_1d, alpha)
    tail = samples_1d[samples_1d <= q]
    return tail.mean()


def cvar_of_negative_total(shortfall, alpha=0.1):
    # flip the sign so the ugly tail is the big shortfalls
    total = region_total_shortfall(shortfall, "H") + region_total_shortfall(shortfall, "S")
    return -cvar(-total, alpha=alpha)


def calibrate_overload_thresholds(h_regime, s_regime, scenario, percentile=0.99,
                                   n_calibration=500_000, seed=999):
    # each scenario gets its own cutoff so "rare" means the same thing
    rng = np.random.default_rng(seed)
    ref = sample_shortfall(h_regime, s_regime, scenario, "gaussian", rng, n_calibration)
    threshold_H = np.quantile(region_total_shortfall(ref, "H"), percentile)
    threshold_S = np.quantile(region_total_shortfall(ref, "S"), percentile)
    return threshold_H, threshold_S


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300_000
    h_regime, s_regime = 2, 2

    print("Self-check 1: gaussian vs student-t moments should basically match.")
    for scenario in SCENARIOS:
        sg = sample_shortfall(h_regime, s_regime, scenario, "gaussian", rng, n)
        st = sample_shortfall(h_regime, s_regime, scenario, "student_t", rng, n)
        mean_diff = np.abs(sg.mean(axis=0) - st.mean(axis=0)).max()
        cov_diff = np.abs(np.cov(sg.T) - np.cov(st.T)).max()
        print(f"  scenario={scenario:20s}  max|mean diff|={mean_diff:.4f}  max|cov diff|={cov_diff:.4f}")

    print("\nSelf-check 2: joint overload at each scenario's own 99th percentile.")
    n_large = 2_000_000
    for scenario in SCENARIOS:
        ref = sample_shortfall(h_regime, s_regime, scenario, "gaussian", rng, n_large)
        threshold_H = np.quantile(region_total_shortfall(ref, "H"), 0.99)
        threshold_S = np.quantile(region_total_shortfall(ref, "S"), 0.99)

        sg = sample_shortfall(h_regime, s_regime, scenario, "gaussian", rng, n_large)
        st = sample_shortfall(h_regime, s_regime, scenario, "student_t", rng, n_large)
        p_gauss = joint_overload_probability(sg, threshold_H, threshold_S)
        p_t = joint_overload_probability(st, threshold_H, threshold_S)
        n_events_g, n_events_t = p_gauss * n_large, p_t * n_large
        cvar_gauss = cvar_of_negative_total(sg)
        cvar_t = cvar_of_negative_total(st)
        ratio = p_t / p_gauss if p_gauss > 0 else float("inf")
        print(f"  scenario={scenario:20s}  (thresholds H>{threshold_H:.2f}, S>{threshold_S:.2f})")
        print(f"    P(joint overload): gaussian={p_gauss:.5f} (n_events={n_events_g:.0f})  "
              f"student_t={p_t:.5f} (n_events={n_events_t:.0f})  ratio={ratio:.2f}x")
        print(f"    CVaR(total shortfall): gaussian={cvar_gauss:.3f}  student_t={cvar_t:.3f}")
