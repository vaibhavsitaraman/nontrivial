import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import torch
import torch.nn as nn
import numpy as np


class ParticleNet(nn.Module):
    # no time, no ode, just spit out a cloud of points
    def __init__(self, dim, n_particles, hidden=64):
        super().__init__()
        self.dim = dim
        self.n_particles = n_particles
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_particles * dim),
        )

    def forward(self, batch_size, device="cpu"):
        dummy = torch.ones(batch_size, 1, device=device)
        out = self.net(dummy)
        return out.view(batch_size, self.n_particles, self.dim)


def sliced_cramer_loss(current_particles, target_particles, n_directions, device="cpu"):
    batch, n, dim = current_particles.shape
    # random unit directions
    u = torch.randn(dim, n_directions, device=device)
    u = u / u.norm(dim=0, keepdim=True)

    theta = current_particles @ u
    phi = target_particles @ u

    # sort each 1d projection and match the order stats
    theta_sorted, _ = torch.sort(theta, dim=1)
    phi_sorted, _ = torch.sort(phi, dim=1)
    return ((theta_sorted - phi_sorted) ** 2).mean()


def train_fixed_distribution_sanity_check(dim=2, n_particles=64, n_steps=3000, batch_size=256,
                                            n_directions=8, lr=1e-3, device="cpu", seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    target_means = torch.tensor([[-2.5, 1.0], [3.0, -1.5]])
    target_cov_a = torch.tensor([[1.0, 0.7], [0.7, 1.0]])
    target_cov_b = torch.tensor([[0.6, -0.4], [-0.4, 0.6]])
    weights_mix = torch.tensor([0.5, 0.5])
    L_a = torch.linalg.cholesky(target_cov_a)
    L_b = torch.linalg.cholesky(target_cov_b)

    def sample_target(n):
        comp = torch.multinomial(weights_mix, n, replacement=True)
        z = torch.randn(n, dim)
        out = torch.zeros(n, dim)
        mask_a = comp == 0
        mask_b = comp == 1
        out[mask_a] = target_means[0] + z[mask_a] @ L_a.T
        out[mask_b] = target_means[1] + z[mask_b] @ L_b.T
        return out

    model = ParticleNet(dim=dim, n_particles=n_particles).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        # every particle is its own draw, don't copy one sample n times
        target_particles = sample_target(batch_size * n_particles).view(batch_size, n_particles, dim).to(device)
        current_particles = model(batch_size, device=device)

        loss = sliced_cramer_loss(current_particles, target_particles, n_directions, device=device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if step % 500 == 0 or step == n_steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    return model, losses


if __name__ == "__main__":
    from scipy.spatial.distance import cdist

    def energy_distance(X, Y, n_sub=1000, seed=0):
        rng = np.random.default_rng(seed)
        X = X[rng.choice(len(X), size=min(n_sub, len(X)), replace=False)]
        Y = Y[rng.choice(len(Y), size=min(n_sub, len(Y)), replace=False)]
        return 2 * cdist(X, Y).mean() - cdist(X, X).mean() - cdist(Y, Y).mean()

    print("sliced-cramer sanity check on a known 2d mixture")
    model, losses = train_fixed_distribution_sanity_check(dim=2, n_particles=64, n_steps=3000)

    with torch.no_grad():
        learned_samples = model(2000).reshape(-1, 2).numpy()

    torch.manual_seed(999)
    target_means = torch.tensor([[-2.5, 1.0], [3.0, -1.5]])
    target_cov_a = torch.tensor([[1.0, 0.7], [0.7, 1.0]])
    target_cov_b = torch.tensor([[0.6, -0.4], [-0.4, 0.6]])
    weights_mix = torch.tensor([0.5, 0.5])
    L_a = torch.linalg.cholesky(target_cov_a)
    L_b = torch.linalg.cholesky(target_cov_b)
    comp = torch.multinomial(weights_mix, 5000, replacement=True)
    z = torch.randn(5000, 2)
    true_samples = torch.zeros(5000, 2)
    true_samples[comp == 0] = target_means[0] + z[comp == 0] @ L_a.T
    true_samples[comp == 1] = target_means[1] + z[comp == 1] @ L_b.T
    true_samples = true_samples.numpy()

    edist = energy_distance(learned_samples, true_samples)
    true_cov = np.cov(true_samples.T)
    learned_cov = np.cov(learned_samples.T)
    true_corr = true_cov[0, 1] / np.sqrt(true_cov[0, 0] * true_cov[1, 1])
    learned_corr = learned_cov[0, 1] / np.sqrt(learned_cov[0, 0] * learned_cov[1, 1])

    print(f"\nEnergy distance (learned vs. true): {edist:.4f}")
    print(f"True correlation: {true_corr:.4f}   Learned correlation: {learned_corr:.4f}")
    print(f"True mean: {true_samples.mean(axis=0)}   Learned mean: {learned_samples.mean(axis=0)}")
    print(f"True cov:\n{true_cov}\nLearned cov:\n{learned_cov}")

    threshold = 0.6
    if edist < threshold:
        print(f"\nPASS: energy distance {edist:.4f} < {threshold}")
    else:
        print(f"\nFAIL: energy distance {edist:.4f} >= {threshold}")
