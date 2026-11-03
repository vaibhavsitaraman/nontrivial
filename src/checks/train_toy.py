import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import os
import pickle
import time
import torch

import envs.toy as toy_env
from nets.ctrl import train_distributional_td_control
from paths import CKPT

# short toy-mdp run, then stash the weights
t0 = time.time()
model, target_model, losses, greedy_history = train_distributional_td_control(
    toy_env, dim=1, n_steps=3000, recompute_greedy_every=3, device="cpu")
print("training took", time.time() - t0, "s")

torch.save(model.state_dict(), os.path.join(CKPT, "toy.pt"))
with open(os.path.join(CKPT, "toy_meta.pkl"), "wb") as f:
    pickle.dump(dict(losses=losses, greedy_history=greedy_history), f)
print("saved toy model")
