from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SE3Pose:
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    translation: np.ndarray


class TensorProjectionNode:
    """Differentiable-friendly SE(3) point-cloud transform helper."""

    @staticmethod
    def construct_se3_matrix(
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
        translation: np.ndarray,
        dtype: np.dtype = np.float64,
    ) -> np.ndarray:
        """Build a 4x4 homogeneous transform using aerospace roll-pitch-yaw order."""
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)

        cx, sx = math.cos(roll), math.sin(roll)
        cy, sy = math.cos(pitch), math.sin(pitch)
        cz, sz = math.cos(yaw), math.sin(yaw)

        r_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=dtype)
        r_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=dtype)
        r_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=dtype)

        rotation = r_z @ r_y @ r_x
        transform = np.eye(4, dtype=dtype)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(translation, dtype=dtype)
        return transform

    @staticmethod
    def transform_point_cloud_tensor(point_tensor, se3_matrix):
        """Transform `(N, 3)` points with a `(4, 4)` SE(3) matrix."""
        try:
            import torch
        except Exception:  # pragma: no cover - optional dependency fallback
            torch = None

        if torch is not None and isinstance(point_tensor, torch.Tensor):
            ones = torch.ones((*point_tensor.shape[:-1], 1), device=point_tensor.device, dtype=point_tensor.dtype)
            homo = torch.cat([point_tensor, ones], dim=-1)
            matrix = se3_matrix if isinstance(se3_matrix, torch.Tensor) else torch.as_tensor(
                se3_matrix, device=point_tensor.device, dtype=point_tensor.dtype
            )
            return (homo @ matrix.T)[..., :3]

        points = np.asarray(point_tensor)
        ones = np.ones((points.shape[0], 1), dtype=points.dtype)
        homo = np.concatenate([points, ones], axis=1)
        matrix = np.asarray(se3_matrix, dtype=points.dtype)
        return (homo @ matrix.T)[:, :3]
