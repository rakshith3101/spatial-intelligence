from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class VoxelGridConfig:
    voxel_size: Tuple[float, float, float] = (0.25, 0.25, 0.25)
    spatial_bounds: Tuple[float, float, float, float, float, float] = (-10.0, 10.0, -10.0, 10.0, -3.0, 3.0)
    max_points_per_voxel: int = 32


class VoxelGridEngine:
    """Convert irregular point clouds into padded voxel tensors."""

    def __init__(self, config: VoxelGridConfig | None = None):
        self.config = config or VoxelGridConfig()
        self.vx, self.vy, self.vz = self.config.voxel_size
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = self.config.spatial_bounds

    def _valid_mask(self, points: np.ndarray) -> np.ndarray:
        return (
            (points[:, 0] >= self.x_min)
            & (points[:, 0] < self.x_max)
            & (points[:, 1] >= self.y_min)
            & (points[:, 1] < self.y_max)
            & (points[:, 2] >= self.z_min)
            & (points[:, 2] < self.z_max)
        )

    def quantize(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        mask = self._valid_mask(points)
        points = points[mask]
        if points.size == 0:
            return points, np.zeros((0, 3), dtype=np.int32)
        voxel_indices = np.floor(
            np.stack(
                [
                    (points[:, 0] - self.x_min) / self.vx,
                    (points[:, 1] - self.y_min) / self.vy,
                    (points[:, 2] - self.z_min) / self.vz,
                ],
                axis=1,
            )
        ).astype(np.int32)
        return points, voxel_indices

    def encode(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return `(M, T, 4)` voxel tensor and `(M, 3)` voxel coordinates."""
        points, coords = self.quantize(points)
        if points.shape[0] == 0:
            return np.zeros((0, self.config.max_points_per_voxel, 4), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

        buckets: Dict[tuple[int, int, int], list[np.ndarray]] = {}
        for point, coord in zip(points, coords):
            key = tuple(int(v) for v in coord)
            buckets.setdefault(key, []).append(point)

        voxel_keys = sorted(buckets.keys())
        voxel_tensor = np.zeros((len(voxel_keys), self.config.max_points_per_voxel, 4), dtype=np.float32)
        voxel_coords = np.zeros((len(voxel_keys), 3), dtype=np.int32)

        for idx, key in enumerate(voxel_keys):
            bucket = np.asarray(buckets[key], dtype=np.float32)
            count = min(bucket.shape[0], self.config.max_points_per_voxel)
            voxel_tensor[idx, :count, : bucket.shape[1]] = bucket[:count]
            voxel_coords[idx] = np.asarray(key, dtype=np.int32)

        return voxel_tensor, voxel_coords

