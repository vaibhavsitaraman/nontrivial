import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import os
import pickle

from envs.cvar_ref import tabular_greedy_cvar_value_iteration
from paths import CKPT

# particle value iteration, no neural net, then dump it
Z_ref, greedy_action_ref, cvar_table_ref, _ = tabular_greedy_cvar_value_iteration(verbose=True)
out = os.path.join(CKPT, "toy_ref.pkl")
with open(out, "wb") as f:
    pickle.dump(dict(Z_ref=Z_ref, greedy_action_ref=greedy_action_ref, cvar_table_ref=cvar_table_ref), f)
print(f"saved {out}")
