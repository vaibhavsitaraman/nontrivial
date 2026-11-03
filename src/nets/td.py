import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import copy
import torch

from nets.flow import ConditionalVelocityNet, ode_sample
import envs.chain as chain


def soft_update(target_model, model, tau):
    # nudge the frozen copy a little toward the live weights
    with torch.no_grad():
        for tp, p in zip(target_model.parameters(), model.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)


def train_distributional_td(
    env=chain,
    dim=1,
    n_steps=8000,
    batch_size=512,
    lr=1e-3,
    gamma=None,
    n_ode_steps_train=20,  # 8 steps quietly underestimates variance
    target_tau=0.01,
    device="cpu",
    seed=0,
):
    if gamma is None:
        gamma = env.GAMMA

    torch.manual_seed(seed)
    rng = __import__("numpy").random.default_rng(seed)

    # live net plus a slow copy we bootstrap from
    model = ConditionalVelocityNet(dim=dim, n_states=env.N_STATES).to(device)
    target_model = copy.deepcopy(model).to(device)
    for p in target_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        # grab a random batch of states
        state_idx_np = rng.integers(0, env.N_STATES, size=batch_size)
        state_idx = torch.tensor(state_idx_np, dtype=torch.long, device=device)

        # one real step of the env
        reward_np, next_state_idx_np = env.sample_transition(state_idx_np, rng)
        if reward_np.ndim == 1:
            reward_np = reward_np[:, None]
        assert reward_np.shape == (batch_size, dim), \
            f"reward shape {reward_np.shape} != expected {(batch_size, dim)}"
        reward = torch.tensor(reward_np, dtype=torch.float32, device=device)
        next_state_idx = torch.tensor(next_state_idx_np, dtype=torch.long, device=device)

        # same noise for the current path and the successor sample
        x0 = torch.randn(batch_size, dim, device=device)
        z_hat = ode_sample(target_model, next_state_idx, n_ode_steps_train, x0=x0, device=device)
        x1 = reward + gamma * z_hat

        # straight-line flow matching target
        t = torch.rand(batch_size, 1, device=device)
        xt = (1 - t) * x0 + t * x1
        target_v = x1 - x0

        pred_v = model(t, xt, state_idx)
        loss = ((pred_v - target_v) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        soft_update(target_model, model, target_tau)

        losses.append(loss.item())
        if step % 1000 == 0 or step == n_steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    return model, losses


if __name__ == "__main__":
    train_distributional_td(n_steps=500)
