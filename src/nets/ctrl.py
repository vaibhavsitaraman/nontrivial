import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import copy
import torch
import numpy as np

from nets.flow import ConditionalVelocityNetSA, ode_sample_sa


def soft_update(target_model, model, tau):
    with torch.no_grad():
        for tp, p in zip(target_model.parameters(), model.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)


def cvar_from_samples(samples, alpha):
    # lower-tail mean, works on a tensor or a numpy array
    if isinstance(samples, torch.Tensor):
        samples = samples.numpy()
    q = np.quantile(samples, alpha)
    tail = samples[samples <= q]
    return float(tail.mean())


def compute_greedy_actions(target_model, n_states, n_actions, alpha, n_cvar_samples=2000,
                            n_ode_steps=20, device="cpu"):
    cvar_table = np.zeros((n_states, n_actions))
    # score every state-action cell
    for s in range(n_states):
        for a in range(n_actions):
            state_idx = torch.full((n_cvar_samples,), s, dtype=torch.long, device=device)
            action_idx = torch.full((n_cvar_samples,), a, dtype=torch.long, device=device)
            samples = ode_sample_sa(target_model, state_idx, action_idx, n_ode_steps, device=device).squeeze(-1)
            cvar_table[s, a] = cvar_from_samples(samples, alpha)
    greedy_action = np.argmax(cvar_table, axis=1)
    return greedy_action, cvar_table


def train_distributional_td_control(
    env,
    dim=1,
    alpha=0.1,
    n_steps=5000,
    batch_size=512,
    lr=1e-3,
    gamma=None,
    n_ode_steps_train=20,
    n_cvar_samples=2000,
    target_tau=0.01,
    recompute_greedy_every=1,
    device="cpu",
    seed=0,
):
    if gamma is None:
        gamma = env.GAMMA

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = ConditionalVelocityNetSA(dim=dim, n_states=env.N_STATES, n_actions=env.N_ACTIONS).to(device)
    target_model = copy.deepcopy(model).to(device)
    for p in target_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    greedy_action_history = []
    greedy_action, cvar_table = None, None

    for step in range(n_steps):
        # refresh which action looks best, but not necessarily every step
        if step % recompute_greedy_every == 0:
            greedy_action, cvar_table = compute_greedy_actions(
                target_model, env.N_STATES, env.N_ACTIONS, alpha,
                n_cvar_samples=n_cvar_samples, n_ode_steps=n_ode_steps_train, device=device)
            greedy_action_history.append(greedy_action.copy())

        # random state and action, off-policy style
        state_idx_np = rng.integers(0, env.N_STATES, size=batch_size)
        action_idx_np = rng.integers(0, env.N_ACTIONS, size=batch_size)
        state_idx = torch.tensor(state_idx_np, dtype=torch.long, device=device)
        action_idx = torch.tensor(action_idx_np, dtype=torch.long, device=device)

        reward_np, next_state_idx_np = env.sample_transition(state_idx_np, action_idx_np, rng)
        if reward_np.ndim == 1:
            reward_np = reward_np[:, None]
        reward = torch.tensor(reward_np, dtype=torch.float32, device=device)
        next_state_idx = torch.tensor(next_state_idx_np, dtype=torch.long, device=device)

        # bootstrap from the greedy action in the next state
        next_greedy_action_np = greedy_action[next_state_idx_np]
        next_greedy_action = torch.tensor(next_greedy_action_np, dtype=torch.long, device=device)

        x0 = torch.randn(batch_size, dim, device=device)
        z_hat = ode_sample_sa(target_model, next_state_idx, next_greedy_action, n_ode_steps_train,
                               x0=x0, device=device)
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
        if step % 1000 == 0 or step == n_steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}  greedy_action={greedy_action}  "
                  f"cvar_table=\n{np.round(cvar_table, 3)}")

    return model, target_model, losses, greedy_action_history


if __name__ == "__main__":
    import envs.toy as toy_env
    train_distributional_td_control(toy_env, dim=1, n_steps=300)
