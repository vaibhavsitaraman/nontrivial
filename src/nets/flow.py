import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import torch
import torch.nn as nn


class ConditionalVelocityNet(nn.Module):
    # little net: given time, the sample, and which state we're in, guess the velocity
    def __init__(self, dim=1, n_states=3, hidden=64):
        super().__init__()
        self.dim = dim
        self.n_states = n_states
        self.net = nn.Sequential(
            nn.Linear(dim + 1 + n_states, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, x, state_idx):
        # smash x, time, and a one-hot state into one vector
        state_onehot = torch.nn.functional.one_hot(state_idx, num_classes=self.n_states).float()
        inp = torch.cat([x, t, state_onehot], dim=-1)
        return self.net(inp)


@torch.no_grad()
def ode_sample(model, state_idx, n_ode_steps, x0=None, device="cpu"):
    n = state_idx.shape[0]
    dim = model.dim
    # start from noise unless the caller already picked a starting point
    if x0 is None:
        x = torch.randn(n, dim, device=device)
    else:
        x = x0.to(device)

    # walk from t=0 to t=1 in equal steps
    dt = 1.0 / n_ode_steps
    for i in range(n_ode_steps):
        t_val = torch.full((n, 1), i * dt, device=device)
        v = model(t_val, x, state_idx)
        x = x + v * dt
    return x


class ConditionalVelocityNetSA(nn.Module):
    # same idea, but the action gets a one-hot too
    def __init__(self, dim=1, n_states=2, n_actions=2, hidden=64):
        super().__init__()
        self.dim = dim
        self.n_states = n_states
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(dim + 1 + n_states + n_actions, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, x, state_idx, action_idx):
        state_onehot = torch.nn.functional.one_hot(state_idx, num_classes=self.n_states).float()
        action_onehot = torch.nn.functional.one_hot(action_idx, num_classes=self.n_actions).float()
        inp = torch.cat([x, t, state_onehot, action_onehot], dim=-1)
        return self.net(inp)


@torch.no_grad()
def ode_sample_sa(model, state_idx, action_idx, n_ode_steps, x0=None, device="cpu"):
    n = state_idx.shape[0]
    dim = model.dim
    if x0 is None:
        x = torch.randn(n, dim, device=device)
    else:
        x = x0.to(device)

    # euler steps, now conditioned on the action as well
    dt = 1.0 / n_ode_steps
    for i in range(n_ode_steps):
        t_val = torch.full((n, 1), i * dt, device=device)
        v = model(t_val, x, state_idx, action_idx)
        x = x + v * dt
    return x
