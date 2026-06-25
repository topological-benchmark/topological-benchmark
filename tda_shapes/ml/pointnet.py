"""A small PointNet regressor for predicting Betti numbers from point clouds.

Vanilla PointNet (no input/feature transform networks): a shared per-point MLP
followed by a global max-pool (the permutation-invariant aggregation) and a
regression head producing ``(b0, b1, b2)``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

RngLike = int | np.random.Generator | None


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PointNetRegressor(nn.Module):
    """Permutation-invariant regressor: shared MLP -> max-pool -> MLP head."""

    def __init__(self, out_dim: int = 3, dropout: float = 0.3):
        super().__init__()
        # Shared per-point feature extractor (applied to each point identically).
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, 3). BatchNorm1d expects (N, C), so flatten points into the
        # batch dimension for the shared encoder, then restore and max-pool.
        b, p, _ = x.shape
        feats = self.encoder(x.reshape(b * p, 3)).reshape(b, p, -1)
        pooled = feats.max(dim=1).values  # global max-pool over points
        return self.head(pooled)


def train_pointnet(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 80,
    batch: int = 32,
    lr: float = 1e-3,
    device: torch.device | str | None = None,
    rng: RngLike = None,
    verbose: bool = False,
) -> PointNetRegressor:
    """Train a :class:`PointNetRegressor` with Adam + MSE loss."""
    device = torch.device(device) if device is not None else _auto_device()
    seed = rng if isinstance(rng, int) else np.random.default_rng(rng).integers(2**31)
    torch.manual_seed(int(seed))

    model = PointNetRegressor(out_dim=y.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    n = len(xt)
    gen = torch.Generator().manual_seed(int(seed))

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            if len(idx) < 2:  # BatchNorm needs >=2 samples
                continue
            opt.zero_grad()
            pred = model(xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  mse={total / n:.4f}")
    return model


@torch.no_grad()
def predict_pointnet(
    model: PointNetRegressor,
    x: np.ndarray,
    *,
    device: torch.device | str | None = None,
    batch: int = 64,
) -> np.ndarray:
    """Predict rounded, non-negative integer Betti numbers, ``(M, 3)``."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    out = []
    for s in range(0, len(xt), batch):
        out.append(model(xt[s : s + batch]).cpu().numpy())
    pred = np.concatenate(out, axis=0)
    return np.clip(np.rint(pred), 0, None).astype(int)
