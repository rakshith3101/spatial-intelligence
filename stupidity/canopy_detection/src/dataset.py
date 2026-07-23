from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CHMDataset(Dataset):
    """Very small demo Dataset that loads the .npz produced by the demo rasterizer.

    Expects arrays: 'image' (H,W,3), 'chm' (H,W), 'mask' (H,W)
    Returns tensors: image (3,H,W), chm (1,H,W), mask (1,H,W)
    """

    def __init__(self, npz_path: str, transform=None):
        data = np.load(npz_path)
        self.image = data["image"].astype(np.float32)
        self.chm = data["chm"].astype(np.float32)
        self.mask = data["mask"].astype(bool)
        self.transform = transform

    def __len__(self):
        return 1  # demo uses single tile

    def __getitem__(self, idx):
        img = self.image
        tgt = self.chm
        mask = self.mask

        # convert to tensors and channel-first
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        tgt_t = torch.from_numpy(np.nan_to_num(tgt, 0.0)).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        if self.transform:
            img_t = self.transform(img_t)

        return img_t, tgt_t, mask_t


class CHMTileFolderDataset(Dataset):
    """Loads many canopy tiles stored as .npz files in a directory.

    Each tile must contain:
      - image: (H, W, C)
      - chm:   (H, W)
      - mask:  (H, W)
    """

    def __init__(self, tiles_dir: str, transform=None):
        self.tiles = sorted(Path(tiles_dir).glob("*.npz"))
        if not self.tiles:
            raise FileNotFoundError(f"No .npz tiles found in {tiles_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        data = np.load(self.tiles[idx])
        img = data["image"].astype(np.float32)
        tgt = data["chm"].astype(np.float32)
        mask = data["mask"].astype(bool)

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        tgt_t = torch.from_numpy(np.nan_to_num(tgt, 0.0)).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        if self.transform:
            img_t = self.transform(img_t)

        return img_t, tgt_t, mask_t


if __name__ == '__main__':
    ds = CHMDataset('data/demo_chm.npz')
    img, chm, mask = ds[0]
    print(img.shape, chm.shape, mask.shape)
