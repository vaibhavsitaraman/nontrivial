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


def compute_greedy_actions_floq(target_model, n_states, n_actions, dim, weights, device="cpu",
                                 n_ode_steps=20, n_point_samples=20):
    point_value_table = np.zeros((n_states, n_actions))
    for s in range(n_states):
        state_idx = torch.full((n_point_samples,), s, dtype=torch.long, device=device)
        for a in range(n_actions):
            action_idx = torch.full((n_point_samples,), a, dtype=torch.long, device=device)
            # average a bunch of starts so one lucky x0 doesn't win the argmax
            x0 = torch.randn(n_point_samples, dim, device=device)
            points = ode_sample_sa(target_model, state_idx, action_idx, n_ode_steps, x0=x0, device=device)
            mean_point = points.mean(dim=0, keepdim=True).cpu().numpy()
            # alpha near 1 on one point is just the point
            point_value_table[s, a] = weighted_cvar(mean_point, weights, alpha=0.99)
    greedy_action = np.argmax(point_value_table, axis=1)
    return greedy_action, point_value_table


def train_floq(
    env,
    dim,
    weights,
    n_steps=5000,
    batch_size=512,
    lr=1e-3,
    gamma=None,
    n_ode_steps_train=25,
    target_tau=0.01,
    recompute_greedy_every=25,
    n_point_samples_greedy=20,
    n_point_samples_target=8,
    device="cpu",
    seed=0,
    log_every=500,
):
    if gamma is None:
        gamma = env.GAMMA
    weights = np.asarray(weights)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = ConditionalVelocityNetSA(dim=dim, n_states=env.N_STATES, n_actions=env.N_ACTIONS).to(device)
    target_model = copy.deepcopy(model).to(device)
    for p in target_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    greedy_action_history = []
    greedy_action, point_value_table = None, None

    for step in range(n_steps):
        if step % recompute_greedy_every == 0:
            greedy_action, point_value_table = compute_greedy_actions_floq(
                target_model, env.N_STATES, env.N_ACTIONS, dim, weights, device=device,
                n_ode_steps=n_ode_steps_train, n_point_samples=n_point_samples_greedy)
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

        # current path still needs random noise; the successor is a point estimate
        x0 = torch.randn(batch_size, dim, device=device)

        with torch.no_grad():
            rep_state = next_state_idx.repeat_interleave(n_point_samples_target)
            rep_action = next_greedy_action.repeat_interleave(n_point_samples_target)
            x0_rep = torch.randn(batch_size * n_point_samples_target, dim, device=device)
            points = ode_sample_sa(target_model, rep_state, rep_action, n_ode_steps_train,
                                    x0=x0_rep, device=device)
            z_hat = points.view(batch_size, n_point_samples_target, dim).mean(dim=1)

        x1 = reward + gamma * z_hat

        t = torch.rand(batch_size, 1, device=device)
        xt = (1 - t) * x0 + t * x1
        target_v = x1 - x0

        pred_v = model(t, xt, state_idx, action_idx)
        loss = ((pred_v - target_v) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        soft_update(target_model, model, target_tau)

        losses.append(loss.item())
        if step % log_every == 0 or step == n_steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}  greedy_action={greedy_action}")

    return model, target_model, losses, greedy_action_history
