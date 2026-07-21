from __future__ import annotations

from torch.utils.data import Dataset

from .synthetic_scene import SyntheticSceneGenerator


class SyntheticThreatSceneDataset(Dataset):
    """Torch dataset wrapper around synthetic labeled threat scenes."""

    def __init__(self, num_scenes: int = 64, seed: int = 7):
        self.generator = SyntheticSceneGenerator(seed=seed)
        self.scenes = self.generator.generate_dataset(num_scenes=num_scenes)

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int):
        scene = self.scenes[idx]
        return scene.points, scene.boxes, scene.class_ids, scene.point_labels, scene.metadata

