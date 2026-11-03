import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

from paths import PLOTS

torch.manual_seed(0)
np.random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_MEANS = torch.tensor([-2.5, 3.0])
TARGET_STDS = torch.tensor([0.6, 1.0])
TARGET_WEIGHTS = torch.tensor([0.4, 0.6])


def sample_target(n):
    # pick a bump, then jitter around it
    comp = torch.multinomial(TARGET_WEIGHTS, n, replacement=True)
    means = TARGET_MEANS[comp]
    stds = TARGET_STDS[comp]
    return (means + stds * torch.randn(n)).unsqueeze(-1)


class VelocityNet(nn.Module):
    def __init__(self, dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, x):
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)


def train(model, n_steps=3000, batch_size=512, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(n_steps):
        x1 = sample_target(batch_size).to(device)
        x0 = torch.randn_like(x1).to(device)
        t = torch.rand(batch_size, 1).to(device)
        xt = (1 - t) * x0 + t * x1
        target_v = x1 - x0

        pred_v = model(t, xt)
        loss = ((pred_v - target_v) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if step % 500 == 0 or step == n_steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")
    return losses


@torch.no_grad()
def sample(model, n_samples, n_ode_steps):
    x = torch.randn(n_samples, 1).to(device)
    dt = 1.0 / n_ode_steps
    for i in range(n_ode_steps):
        t_val = torch.full((n_samples, 1), i * dt, device=device)
        v = model(t_val, x)
        x = x + v * dt
    return x.cpu()


def main():
    model = VelocityNet(dim=1).to(device)
    print("training a plain flow on a 1d mixture")
    train(model, n_steps=3000)

    true_samples = sample_target(5000).squeeze(-1).numpy()

    # see how many euler steps you actually need
    step_counts = [1, 2, 4, 8, 20, 50]
    results = {}
    for n_steps_ode in step_counts:
        gen = sample(model, 5000, n_steps_ode).squeeze(-1).numpy()
        w1 = wasserstein_distance(gen, true_samples)
        results[n_steps_ode] = (gen, w1)
        print(f"ODE steps={n_steps_ode:3d}   Wasserstein-1 = {w1:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.hist(true_samples, bins=60, density=True, alpha=0.5, label="ground truth", color="tab:blue")
    best_gen, _ = results[step_counts[-1]]
    ax.hist(best_gen, bins=60, density=True, alpha=0.5,
             label=f"flow-matched ({step_counts[-1]} ODE steps)", color="tab:orange")
    ax.set_title("Learned vs. true 1D distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend()

    ax = axes[1]
    xs = list(results.keys())
    ys = [results[k][1] for k in xs]
    ax.plot(xs, ys, marker="o", color="tab:red")
    ax.set_xscale("log")
    ax.set_title("Sample fidelity vs. ODE integration steps")
    ax.set_xlabel("number of ODE steps (log scale)")
    ax.set_ylabel("Wasserstein-1 distance to ground truth")

    plt.tight_layout()
    out_path = os.path.join(PLOTS, "flow_check.png")
    plt.savefig(out_path, dpi=150)
    print(f"saved {out_path}")

    final_w1 = results[step_counts[-1]][1]
    threshold = 0.15
    if final_w1 < threshold:
        print(f"PASS: W1 at {step_counts[-1]} steps = {final_w1:.4f} < {threshold}")
    else:
        print(f"FAIL: W1 at {step_counts[-1]} steps = {final_w1:.4f} >= {threshold}")


if __name__ == "__main__":
    main()
