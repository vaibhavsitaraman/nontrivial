import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import copy
import torch
import torch.nn as nn
import numpy as np

from nets.metrics import weighted_cvar
from nets.particles import sliced_cramer_loss


class ParticleNetSA(nn.Module):
    def __init__(self, dim, n_states, n_actions, n_particles, hidden=64):
        super().__init__()
        self.dim = dim
        self.n_particles = n_particles
        self.n_states = n_states
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(n_states + n_actions, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_particles * dim),
        )

    def forward(self, state_idx, action_idx):
        state_onehot = torch.nn.functional.one_hot(state_idx, num_classes=self.n_states).float()
        action_onehot = torch.nn.functional.one_hot(action_idx, num_classes=self.n_actions).float()
        inp = torch.cat([state_onehot, action_onehot], dim=-1)
        out = self.net(inp)
        return out.view(-1, self.n_particles, self.dim)


def soft_update(target_model, model, tau):
    with torch.no_grad():
        for tp, p in zip(target_model.parameters(), model.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)


def compute_greedy_actions_sliced_cramer(target_model, n_states, n_actions, weights, alpha, device="cpu"):
    # one forward pass is the whole sample cloud
    cvar_table = np.zeros((n_states, n_actions))
    with torch.no_grad():
        for s in range(n_states):
            for a in range(n_actions):
                state_idx = torch.full((1,), s, dtype=torch.long, device=device)
                action_idx = torch.full((1,), a, dtype=torch.long, device=device)
                particles = target_model(state_idx, action_idx).squeeze(0).cpu().numpy()
                cvar_table[s, a] = weighted_cvar(particles, weights, alpha=alpha)
    greedy_action = np.argmax(cvar_table, axis=1)
    return greedy_action, cvar_table


def train_sliced_cramer_control(
    env,
    dim,
    weights,
    alpha=0.1,
    n_particles=192,
    n_directions=16,
    n_steps=5000,
    batch_size=512,
    lr=1e-3,
    gamma=None,
    target_tau=0.01,
    recompute_greedy_every=25,
    device="cpu",
    seed=0,
    log_every=500,
):
    if gamma is None:
        gamma = env.GAMMA
    weights = np.asarray(weights)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = ParticleNetSA(dim=dim, n_states=env.N_STATES, n_actions=env.N_ACTIONS,
                           n_particles=n_particles).to(device)
    target_model = copy.deepcopy(model).to(device)
    for p in target_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    greedy_action_history = []
    greedy_action, cvar_table = None, None

    for step in range(n_steps):
        if step % recompute_greedy_every == 0:
            greedy_action, cvar_table = compute_greedy_actions_sliced_cramer(
                target_model, env.N_STATES, env.N_ACTIONS, weights, alpha, device=device)
            greedy_action_history.append(greedy_action.copy())

        state_idx_np = rng.integers(0, env.N_STATES, size=batch_size)
        action_idx_np = rng.integers(0, env.N_ACTIONS, size=batch_size)
        state_idx = torch.tensor(state_idx_np, dtype=torch.long, device=device)
        action_idx = torch.tensor(action_idx_np, dtype=torch.long, device=device)

        reward_np, next_state_idx_np = env.sample_transition(state_idx_np, action_idx_np, rng)
        reward = torch.tensor(reward_np, dtype=torch.float32, device=device)
        next_state_idx = torch.tensor(next_state_idx_np, dtype=torch.long, device=device)

        next_greedy_action_np = greedy_action[next_state_idx_np]
        next_greedy_action = torch.tensor(next_greedy_action_np, dtype=torch.long, device=device)

        with torch.no_grad():
            target_successor_particles = target_model(next_state_idx, next_greedy_action)
            target_particles = reward.unsqueeze(1) + gamma * target_successor_particles

        current_particles = model(state_idx, action_idx)
        loss = sliced_cramer_loss(current_particles, target_particles, n_directions, device=device)

        opt.zero_grad()
        loss.backward()
        opt.step()
        soft_update(target_model, model, target_tau)

        losses.append(loss.item())
        if step % log_every == 0 or step == n_steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}  greedy_action={greedy_action}")

    return model, target_model, losses, greedy_action_history
