import os

# project root sits one folder above src
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT = os.path.join(ROOT, "checkpoints")
PLOTS = os.path.join(ROOT, "plots")
DATA = os.path.join(ROOT, "data")

# make the buckets if a fresh checkout doesn't have them yet
for folder in (CKPT, PLOTS, DATA):
    os.makedirs(folder, exist_ok=True)
