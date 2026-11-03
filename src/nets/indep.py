import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import copy
import torch
import numpy as np

from nets.flow import ConditionalVelocityNetSA, ode_sample_sa
from nets.metrics import weighted_cvar


def soft_update(target_model, model, tau):
    with torch.no_grad():
        for tp, p in zip(target_model.parameters(), model.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)


def sample_independent_combined(target_models, state_idx, action_idx, n_ode_steps, device="cpu"):
    # each dim gets its own noise, so they can't learn to move together
    n = state_idx.shape[0]
    dim = len(target_models)
    out = torch.zeros(n, dim, device=device)
    for d, model_d in enumerate(target_models):
        s_d = ode_sample_sa(model_d, state_idx, action_idx, n_ode_steps, device=device)
        out[:, d] = s_d.squeeze(-1)
    return out


def compute_greedy_actions_independent(target_models, n_states, n_actions, weights, alpha,
                                          n_cvar_samples=500, n_ode_steps=10, device="cpu"):
    cvar_table = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            state_idx = torch.full((n_cvar_samples,), s, dtype=torch.long, device=device)
            action_idx = torch.full((n_cvar_samples,), a, dtype=torch.long, device=device)
            samples = sample_independent_combined(target_models, state_idx, action_idx, n_ode_steps, device=device)
            samples_np = samples.cpu().numpy()
            cvar_table[s, a] = weighted_cvar(samples_np, weights, alpha=alpha)
    greedy_action = np.argmax(cvar_table, axis=1)
    return greedy_action, cvar_table


def train_independent_baseline(
    env,
    dim,
    weights,
    alpha=0.1,
    n_steps=5000,
    batch_size=512,
    lr=1e-3,
    gamma=None,
    n_ode_steps_train=10,
    n_cvar_samples=500,
    target_tau=0.01,
    recompute_greedy_every=10,
    device="cpu",
    seed=0,
    log_every=500,
):
    if gamma is None:
        gamma = env.GAMMA
    weights = np.asarray(weights)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # one tiny 1d net per reward dimension
    models = [ConditionalVelocityNetSA(dim=1, n_states=env.N_STATES, n_actions=env.N_ACTIONS).to(device)
              for _ in range(dim)]
    target_models = [copy.deepcopy(m).to(device) for m in models]
    for tm in target_models:
        for p in tm.parameters():
            p.requires_grad_(False)

    opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in models]
    losses = [[] for _ in range(dim)]
    greedy_action_history = []
    greedy_action, cvar_table = None, None

    for step in range(n_steps):
        if step % recompute_greedy_every == 0:
            greedy_action, cvar_table = compute_greedy_actions_independent(
                target_models, env.N_STATES, env.N_ACTIONS, weights, alpha,
                n_cvar_samples=n_cvar_samples, n_ode_steps=n_ode_steps_train, device=device)
            greedy_action_history.append(greedy_action.copy())

        state_idx_np = rng.integers(0, env.N_STATES, size=batch_size)
        action_idx_np = rng.integers(0, env.N_ACTIONS, size=batch_size)
        state_idx = torch.tensor(state_idx_np, dtype=torch.long, device=device)
        action_idx = torch.tensor(action_idx_np, dtype=torch.long, device=device)

        # same transition for everyone, but each net only sees its own column
        reward_np, next_state_idx_np = env.sample_transition(state_idx_np, action_idx_np, rng)
        reward = torch.tensor(reward_np, dtype=torch.float32, device=device)
        next_state_idx = torch.tensor(next_state_idx_np, dtype=torch.long, device=device)

        next_greedy_action_np = greedy_action[next_state_idx_np]
        next_greedy_action = torch.tensor(next_greedy_action_np, dtype=torch.long, device=device)

        step_losses = []
        for d in range(dim):
            x0_d = torch.randn(batch_size, 1, device=device)
            z_hat_d = ode_sample_sa(target_models[d], next_state_idx, next_greedy_action, n_ode_steps_train,
                                     x0=x0_d, device=device)
            x1_d = reward[:, d:d+1] + gamma * z_hat_d

            t = torch.rand(batch_size, 1, device=device)
            xt_d = (1 - t) * x0_d + t * x1_d
            target_v_d = x1_d - x0_d

            pred_v_d = models[d](t, xt_d, state_idx, action_idx)
            loss_d = ((pred_v_d - target_v_d) ** 2).mean()

            opts[d].zero_grad()
            loss_d.backward()
            opts[d].step()
            soft_update(target_models[d], models[d], target_tau)

            losses[d].append(loss_d.item())
            step_losses.append(loss_d.item())

        if step % log_every == 0 or step == n_steps - 1:
            print(f"step {step:5d}  losses(per-dim)={[round(l, 3) for l in step_losses]}  "
                  f"greedy_action={greedy_action}")

    return models, target_models, losses, greedy_action_history
