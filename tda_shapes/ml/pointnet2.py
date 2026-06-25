"""PointNet++ (single-scale grouping) classifier for Betti-number prediction.

A standard PointNet++ SSG architecture (Qi et al., 2017): two
:class:`PointNetSetAbstraction` layers with farthest-point sampling + ball-query
grouping + a local mini-PointNet, then a global set-abstraction layer, then one
softmax head per Betti dimension. Unlike vanilla PointNet's single global
max-pool, the hierarchical set abstraction aggregates *local* neighborhoods, the
inductive bias needed to capture topology-related structure.

The sampling/grouping helpers are the canonical pure-PyTorch implementations
(no compiled CUDA ops), so the model runs unchanged on CPU, MPS and CUDA. The
group radii default to values scaled for the unit-RMS normalized clouds this
project uses (see :func:`tda_shapes.ml.data.normalize_cloud`) rather than the
unit-ball defaults of the original paper.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance between two point sets.

    ``src`` is ``(B, N, C)``, ``dst`` is ``(B, M, C)``; returns ``(B, N, M)``.
    """
    b, n, _ = src.shape
    _, m, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(b, n, 1)
    dist += torch.sum(dst**2, -1).view(b, 1, m)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index. ``points`` ``(B, N, C)``, ``idx`` ``(B, ...)``."""
    device = points.device
    b = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(b, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest-point sampling: indices ``(B, npoint)`` of well-spread points."""
    device = xyz.device
    b, n, _ = xyz.shape
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=device)
    distance = torch.ones(b, n, device=device) * 1e10
    farthest = torch.randint(0, n, (b,), dtype=torch.long, device=device)
    batch_indices = torch.arange(b, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(b, 1, -1)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """Group up to ``nsample`` neighbours within ``radius`` of each query point."""
    device = xyz.device
    b, n, _ = xyz.shape
    _, s, _ = new_xyz.shape
    group_idx = (
        torch.arange(n, dtype=torch.long, device=device).view(1, 1, n).repeat([b, s, 1])
    )
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = n
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # Points beyond the radius were sorted to the end as `n`; backfill them with
    # the nearest in-radius neighbour (the first column) so groups stay full.
    group_first = group_idx[:, :, 0].view(b, s, 1).repeat([1, 1, nsample])
    mask = group_idx == n
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: torch.Tensor | None,
):
    """FPS centroids + ball-query groups, with grouped xyz recentred locally."""
    b, _, c = xyz.shape
    npoint = min(npoint, xyz.shape[1])  # guard small clouds
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) - new_xyz.view(b, npoint, 1, c)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group_all(xyz: torch.Tensor, points: torch.Tensor | None):
    """Treat the whole cloud as one group (the final global layer)."""
    device = xyz.device
    b, n, c = xyz.shape
    new_xyz = torch.zeros(b, 1, c, device=device)
    grouped_xyz = xyz.view(b, 1, n, c)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(b, 1, n, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """One SSG set-abstraction layer: sample -> group -> shared MLP -> max-pool."""

    def __init__(
        self,
        npoint: int | None,
        radius: float | None,
        nsample: int | None,
        in_channel: int,
        mlp: Sequence[int],
        group_all: bool,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        # xyz: (B, C, N), points: (B, D, N) -- channels-first, as in the reference.
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        # new_points: (B, npoint, nsample, C+D) -> (B, C+D, nsample, npoint)
        new_points = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]  # max-pool over the local group
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNet2Classifier(nn.Module):
    """PointNet++ SSG with one softmax head per Betti dimension.

    ``n_classes`` gives the number of categories for each Betti dimension;
    :meth:`forward` returns a list of per-dimension logit tensors, matching
    :class:`tda_shapes.ml.pointnet.PointNetClassifier` so the two are
    interchangeable in the training loop and benchmark.

    The default group radii suit unit-RMS clouds; widen/narrow them (or the
    ``npoint``/``nsample`` counts) if you change the normalization or point count.
    """

    def __init__(
        self,
        n_classes: Sequence[int],
        *,
        npoint1: int = 128,
        radius1: float = 0.4,
        nsample1: int = 32,
        npoint2: int = 32,
        radius2: float = 0.8,
        nsample2: int = 32,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.n_classes = list(n_classes)
        # +3 at each level for the locally-recentred grouped xyz coordinates.
        self.sa1 = PointNetSetAbstraction(
            npoint1, radius1, nsample1, in_channel=3, mlp=[64, 64, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint2, radius2, nsample2, in_channel=128 + 3, mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            None, None, None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )
        self.trunk = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList(nn.Linear(256, c) for c in self.n_classes)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # x: (B, N, 3) -> channels-first xyz (B, 3, N); no extra point features.
        b = x.shape[0]
        xyz = x.permute(0, 2, 1)
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        h = self.trunk(l3_points.view(b, 1024))
        return [head(h) for head in self.heads]
