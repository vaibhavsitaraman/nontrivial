import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import os
import pickle
import torch
import numpy as np

import envs.toy as env
from nets.flow import ConditionalVelocityNetSA, ode_sample_sa
from nets.ctrl import cvar_from_samples
from paths import CKPT

EVAL_SAMPLES = 8000
EVAL_ODE_STEPS = 50

with open(os.path.join(CKPT, "toy_ref.pkl"), "rb") as f:
    ref = pickle.load(f)
Z_ref = ref["Z_ref"]
greedy_action_ref = ref["greedy_action_ref"]

model = ConditionalVelocityNetSA(dim=1, n_states=env.N_STATES, n_actions=env.N_ACTIONS)
model.load_state_dict(torch.load(os.path.join(CKPT, "toy.pt"), map_location="cpu"))
model.eval()

print("learned model vs the tabular reference")
all_pass = True
for s in range(env.N_STATES):
    # cvar of every action, then pick the best one
    state_idx_all = torch.full((EVAL_SAMPLES,), s, dtype=torch.long)
    cvars = []
    for a in range(env.N_ACTIONS):
        action_idx_all = torch.full((EVAL_SAMPLES,), a, dtype=torch.long)
        samples = ode_sample_sa(model, state_idx_all, action_idx_all, EVAL_ODE_STEPS).squeeze(-1).numpy()
        cvars.append(cvar_from_samples(samples, env.CVAR_ALPHA))
    learned_greedy = int(np.argmax(cvars))

    action_idx_greedy = torch.full((EVAL_SAMPLES,), learned_greedy, dtype=torch.long)
    learned_samples = ode_sample_sa(model, state_idx_all, action_idx_greedy, EVAL_ODE_STEPS).squeeze(-1).numpy()

    z_ref = Z_ref[(s, greedy_action_ref[s])]
    policy_match = learned_greedy == greedy_action_ref[s]
    mean_err = abs(learned_samples.mean() - z_ref.mean())
    var_err = abs(learned_samples.var() - z_ref.var())
    cvar_err = abs(cvar_from_samples(learned_samples, env.CVAR_ALPHA) - cvar_from_samples(z_ref, env.CVAR_ALPHA))

    print(f"\n  state {env.STATE_NAMES[s]}:")
    print(f"    tabular={env.ACTION_NAMES[greedy_action_ref[s]]}  learned={env.ACTION_NAMES[learned_greedy]}  "
          f"{'MATCH' if policy_match else 'MISMATCH'}")
    print(f"    errors: mean={mean_err:.3f}  var={var_err:.3f}  CVaR={cvar_err:.3f}")

    state_pass = policy_match and mean_err < 0.8 and var_err < 0.8 and cvar_err < 1.2
    all_pass &= state_pass
    print(f"    {'PASS' if state_pass else 'FAIL'}")

print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
