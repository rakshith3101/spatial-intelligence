from __future__ import annotations

import torch
import torch.nn as nn


class VoxelFeatureEncoder(nn.Module):
    """Point-wise VFE layer with centroid offsets and max pooling."""

    def __init__(self, in_channels: int = 4, hidden_dim: int = 32, out_dim: int = 16):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels + 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxels: `(M, T, 4)` tensor with zero-padded points.
        Returns:
            `(M, 16)` voxel embeddings.
        """
        if voxels.ndim != 3:
            raise ValueError(f"Expected `(M, T, C)` tensor, got {tuple(voxels.shape)}")

        xyz = voxels[..., :3]
        mask = (voxels.abs().sum(dim=-1, keepdim=True) > 0).float()
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        centroid = (xyz * mask).sum(dim=1, keepdim=True) / denom
        offsets = xyz - centroid
        features = torch.cat([voxels, offsets], dim=-1)

        m, t, c = features.shape
        flattened = features.reshape(m * t, c)
        encoded = self.point_mlp(flattened).reshape(m, t, -1)
        encoded = encoded * mask
        pooled = encoded.max(dim=1).values
        pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
        return pooled

