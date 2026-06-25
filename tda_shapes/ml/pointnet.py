"""A small PointNet classifier for predicting Betti numbers from point clouds.

Vanilla PointNet (no input/feature transform networks): a shared per-point MLP
followed by a global max-pool (the permutation-invariant aggregation) and one
softmax classification head per Betti dimension, so ``(b0, b1, b2)`` are
predicted as independent categorical labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from torch import nn

RngLike = int | np.random.Generator | None
Arch = Literal["pointnet", "pointnet2"]


def _build_model(arch: Arch, n_classes: Sequence[int]) -> nn.Module:
    """Construct a multi-head point-cloud classifier for the given architecture."""
    if arch == "pointnet":
        return PointNetClassifier(n_classes=n_classes)
    if arch == "pointnet2":
        from .pointnet2 import PointNet2Classifier

        return PointNet2Classifier(n_classes=n_classes)
    raise ValueError(f"Unknown architecture: {arch!r}")


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PointNetClassifier(nn.Module):
    """Permutation-invariant classifier: shared MLP -> max-pool -> per-dim heads.

    ``n_classes`` gives the number of categories for each Betti dimension; the
    network has one independent softmax head per entry and :meth:`forward`
    returns a list of per-dimension logit tensors.
    """

    def __init__(self, n_classes: Sequence[int], dropout: float = 0.3):
        super().__init__()
        self.n_classes = list(n_classes)
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
        self.trunk = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # One classification head per Betti dimension.
        self.heads = nn.ModuleList(nn.Linear(128, c) for c in self.n_classes)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # x: (B, P, 3). BatchNorm1d expects (N, C), so flatten points into the
        # batch dimension for the shared encoder, then restore and max-pool.
        b, p, _ = x.shape
        feats = self.encoder(x.reshape(b * p, 3)).reshape(b, p, -1)
        pooled = feats.max(dim=1).values  # global max-pool over points
        h = self.trunk(pooled)
        return [head(h) for head in self.heads]  # per-dim logits, each (B, C_i)


@torch.no_grad()
def _exact_acc(
    model: nn.Module, xt: torch.Tensor, yt: torch.Tensor, batch: int = 256
) -> float:
    """Exact-match accuracy (all Betti dims correct) over ``(xt, yt)`` in eval mode."""
    was_training = model.training
    model.eval()
    preds = []
    for s in range(0, len(xt), batch):
        logits = model(xt[s : s + batch])
        preds.append(torch.stack([lg.argmax(dim=1) for lg in logits], dim=1))
    if was_training:
        model.train()
    pred = torch.cat(preds, dim=0)
    return float((pred == yt).all(dim=1).float().mean())


def train_pointnet(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 80,
    batch: int = 32,
    lr: float = 1e-3,
    n_classes: Sequence[int] | None = None,
    arch: Arch = "pointnet",
    device: torch.device | str | None = None,
    rng: RngLike = None,
    val: tuple[np.ndarray, np.ndarray] | None = None,
    verbose: bool = False,
) -> nn.Module:
    """Train a point-cloud classifier with Adam + summed cross-entropy.

    ``arch`` selects the model: ``"pointnet"`` (vanilla, global max-pool) or
    ``"pointnet2"`` (PointNet++ hierarchical set abstraction). Both expose the
    same multi-head interface, so the training loop is identical.

    ``y`` holds integer Betti labels, shape ``(N, D)``. ``n_classes`` (one entry
    per Betti dimension) defaults to ``y.max(axis=0) + 1``, i.e. the categories
    observed in the training labels.

    Pass ``val=(x_val, y_val)`` to monitor a held-out split: when ``verbose`` the
    logged epochs report train and validation exact-match accuracy, making
    train/val divergence (overfitting) visible during training.
    """
    device = torch.device(device) if device is not None else _auto_device()
    seed = rng if isinstance(rng, int) else np.random.default_rng(rng).integers(2**31)
    torch.manual_seed(int(seed))

    y_int = np.rint(np.asarray(y)).astype(np.int64)
    if n_classes is None:
        n_classes = (y_int.max(axis=0) + 1).tolist()

    model = _build_model(arch, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_int, dtype=torch.long, device=device)
    n = len(xt)
    n_dims = yt.shape[1]
    gen = torch.Generator().manual_seed(int(seed))

    xv = yv = None
    if val is not None:
        xv = torch.as_tensor(val[0], dtype=torch.float32, device=device)
        yv = torch.as_tensor(
            np.rint(np.asarray(val[1])).astype(np.int64), dtype=torch.long, device=device
        )

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            if len(idx) < 2:  # BatchNorm needs >=2 samples
                continue
            opt.zero_grad()
            logits = model(xt[idx])
            loss = sum(loss_fn(logits[d], yt[idx, d]) for d in range(n_dims))
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            msg = f"  epoch {epoch:3d}  ce={total / n:.4f}  train_acc={_exact_acc(model, xt, yt):.3f}"
            if xv is not None:
                msg += f"  val_acc={_exact_acc(model, xv, yv):.3f}"
            print(msg)
    return model


@torch.no_grad()
def predict_pointnet(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device | str | None = None,
    batch: int = 64,
) -> np.ndarray:
    """Predict integer Betti labels, ``(M, D)``, via per-dimension argmax."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    out = []
    for s in range(0, len(xt), batch):
        logits = model(xt[s : s + batch])
        preds = torch.stack([lg.argmax(dim=1) for lg in logits], dim=1)
        out.append(preds.cpu().numpy())
    return np.concatenate(out, axis=0).astype(int)
