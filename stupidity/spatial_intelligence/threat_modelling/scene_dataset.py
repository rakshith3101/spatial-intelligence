from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from .pygame_scene import PygameThreatWorld


class SyntheticVoxelDataset(Dataset):
    """Dataset of voxel-ready synthetic scenes generated from the Pygame world."""

    def __init__(self, num_samples: int = 1000, seed: int = 7):
        self.world = PygameThreatWorld(seed=seed)
        self.samples = [self._make_sample() for _ in range(num_samples)]

    def _sample_points_from_object(self, obj, n_points: int = 128) -> np.ndarray:
        center = obj["center"]
        size = obj["size"]
        spread = np.maximum(size / 4.0, 0.05)
        xyz = center + np.random.default_rng().normal(scale=spread, size=(n_points, 3)).astype(np.float32)
        intensity = np.full((n_points, 1), 0.75 if obj["threat"] else 0.35, dtype=np.float32)
        return np.hstack([xyz, intensity])

    def _make_sample(self):
        scene = self.world.step()
        points = []
        boxes = []
        labels = []
        for obj in scene["objects"]:
            points.append(self._sample_points_from_object(obj))
            boxes.append(np.array([*obj["center"], *obj["size"]], dtype=np.float32))
            labels.append(int(obj["threat"]))
        background = np.random.default_rng().uniform(-10, 10, size=(256, 3)).astype(np.float32)
        background_i = np.random.default_rng().uniform(0.0, 0.2, size=(256, 1)).astype(np.float32)
        points.append(np.hstack([background, background_i]))
        points = np.concatenate(points, axis=0)
        return {
            "points": points.astype(np.float32),
            "boxes": np.stack(boxes, axis=0).astype(np.float32),
            "labels": np.asarray(labels, dtype=np.int64),
            "scene": scene,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

