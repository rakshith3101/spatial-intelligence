from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class Phase1CanopyDataset(Dataset):
    """Loads aligned Sentinel/GEDI tiles for canopy height training.

    Expected .npz keys:
      - image: (H, W, C) with C >= 4
      - chm: (H, W)
      - mask: (H, W)
    """

    def __init__(self, tiles_dir: str, transform=None):
        self.tiles = sorted(Path(tiles_dir).glob("*.npz"))
        if not self.tiles:
            raise FileNotFoundError(f"No .npz tiles found in {tiles_dir}")
        self.transform = transform
        self.chm_max = 60.0

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        data = np.load(self.tiles[idx])
        img = data["image"].astype(np.float32)
        chm = data["chm"].astype(np.float32)
        mask = data["mask"].astype(bool)

        img = np.nan_to_num(img, 0.0)
        chm = np.clip(np.nan_to_num(chm, 0.0), 0.0, self.chm_max) / self.chm_max

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        chm_t = torch.from_numpy(chm).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        if self.transform:
            img_t = self.transform(img_t)

        return img_t, chm_t, mask_t
