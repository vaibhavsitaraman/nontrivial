import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import numpy as np
from types import SimpleNamespace

import envs.regimes as regimes
import envs.rewards as copula
import envs.surge as combined

N_STATES = combined.N_STATES
GAMMA = combined.GAMMA
N_DIM = copula.N_DIM

DIRECTIONS = ["none", "H_to_S", "S_to_H"]
INTENSITIES = ["conservative", "aggressive"]
ACTIONS = [(d, i) for d in DIRECTIONS for i in INTENSITIES]
N_ACTIONS = len(ACTIONS)
ACTION_NAMES = [f"{d}/{i}" for d, i in ACTIONS]

MAX_SEVERITY = max(copula.REGIME_SEVERITY.values())
BASE_TRANSFER_EFFECT = {"conservative": 1.0, "aggressive": 2.2}
SOURCE_RISK_MULTIPLIER = {"conservative": 1.0, "aggressive": 1.6}
MEAN_FLOOR = 0.05


def apply_action_to_shortfall(shortfall, h, s, direction, intensity, rng, n):
    if direction == "none":
        return shortfall

    if direction == "H_to_S":
        source_regime, dest_region = h, "S"
    else:
        source_regime, dest_region = s, "H"

    # a peak source has nothing spare to send
    source_capacity_frac = 1.0 - copula.REGIME_SEVERITY[source_regime] / MAX_SEVERITY
    effect = BASE_TRANSFER_EFFECT[intensity] * source_capacity_frac

    # skip the floor when the transfer is basically zero, or it nudges the mean up for free
    if effect > 1e-9:
        dest_idx = [copula.dim_index(dest_region, b) for b in copula.BED_TYPES]
        shortfall[:, dest_idx] = np.maximum(shortfall[:, dest_idx] - effect, MEAN_FLOOR)

    if intensity == "aggressive":
        source_region = "H" if dest_region == "S" else "S"
        source_idx = [copula.dim_index(source_region, b) for b in copula.BED_TYPES]
        extra_risk = rng.normal(0, copula.IDIO_STD[source_idx] * (SOURCE_RISK_MULTIPLIER["aggressive"] - 1.0),
                                 size=(n, len(source_idx)))
        shortfall[:, source_idx] += np.abs(extra_risk)

    return shortfall


def make_control_env(spillover_strength, scenario, condition):
    Q = regimes.build_joint_transition_matrix(spillover_strength)

    def sample_transition(state_idx, action_idx, rng):
        batch = state_idx.shape[0]
        reward = np.zeros((batch, N_DIM))
        next_state_idx = np.empty(batch, dtype=np.int64)

        for st in range(N_STATES):
            h, s = combined.state_to_regimes(st)
            for a_idx, (direction, intensity) in enumerate(ACTIONS):
                mask = (state_idx == st) & (action_idx == a_idx)
                n = int(mask.sum())
                if n == 0:
                    continue
                shortfall = copula.sample_shortfall(h, s, scenario, condition, rng, n)
                shortfall = apply_action_to_shortfall(shortfall, h, s, direction, intensity, rng, n)
                # learners want higher=better, shortfall is the opposite
                reward[mask] = -shortfall
                next_state_idx[mask] = rng.choice(N_STATES, size=n, p=Q[st])

        return reward, next_state_idx

    return SimpleNamespace(N_STATES=N_STATES, N_ACTIONS=N_ACTIONS, GAMMA=GAMMA,
                            sample_transition=sample_transition, Q=Q,
                            spillover_strength=spillover_strength, scenario=scenario, condition=condition)


if __name__ == "__main__":
    print("Self-check: action effects at H=Peak, S=Rising, pandemic, gaussian.")
    env = make_control_env(spillover_strength=0.7, scenario="pandemic", condition="gaussian")
    rng = np.random.default_rng(0)
    h, s = 2, 1
    st = h * regimes.N_REGIMES + s
    n = 20000

    for a_idx, (direction, intensity) in enumerate(ACTIONS):
        state_idx = np.full(n, st, dtype=np.int64)
        action_idx = np.full(n, a_idx, dtype=np.int64)
        reward, _ = env.sample_transition(state_idx, action_idx, rng)
        shortfall = -reward
        total_H = copula.region_total_shortfall(shortfall, "H")
        total_S = copula.region_total_shortfall(shortfall, "S")
        print(f"  action={ACTION_NAMES[a_idx]:20s}  mean shortfall H={total_H.mean():.3f}  "
              f"S={total_S.mean():.3f}  std H={total_H.std():.3f}  std S={total_S.std():.3f}")
