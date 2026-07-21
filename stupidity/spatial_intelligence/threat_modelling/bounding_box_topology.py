from __future__ import annotations

import numpy as np


class BoundingBoxTopology:
    """Utilities for 3D axis-aligned IoU and box evaluation."""

    @staticmethod
    def _corners(box: np.ndarray) -> tuple[float, float, float, float, float, float]:
        x, y, z, w, l, h = box
        return x - w / 2, x + w / 2, y - l / 2, y + l / 2, z - h / 2, z + h / 2

    def calculate_axis_aligned_3d_iou(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        ax_min, ax_max, ay_min, ay_max, az_min, az_max = self._corners(np.asarray(box_a, dtype=np.float64))
        bx_min, bx_max, by_min, by_max, bz_min, bz_max = self._corners(np.asarray(box_b, dtype=np.float64))

        inter_x = max(0.0, min(ax_max, bx_max) - max(ax_min, bx_min))
        inter_y = max(0.0, min(ay_max, by_max) - max(ay_min, by_min))
        inter_z = max(0.0, min(az_max, bz_max) - max(az_min, bz_min))
        intersection = inter_x * inter_y * inter_z

        vol_a = max(0.0, (ax_max - ax_min) * (ay_max - ay_min) * (az_max - az_min))
        vol_b = max(0.0, (bx_max - bx_min) * (by_max - by_min) * (bz_max - bz_min))
        union = vol_a + vol_b - intersection
        return float(intersection / union) if union > 0 else 0.0

    def iou_tensor(self, box_a, box_b):
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency fallback
            raise RuntimeError("Torch is unavailable in this environment.") from exc

        if box_a.shape[-1] != 6 or box_b.shape[-1] != 6:
            raise ValueError("IoU expects boxes in `[x, y, z, w, l, h]` format.")
        box_a_np = box_a.detach().cpu().numpy()
        box_b_np = box_b.detach().cpu().numpy()
        return torch.tensor(self.calculate_axis_aligned_3d_iou(box_a_np, box_b_np), dtype=torch.float64, device=box_a.device)
