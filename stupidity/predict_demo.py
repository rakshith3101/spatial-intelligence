from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from canopy_detection.src.dataset import CHMDataset, CHMTileFolderDataset
from canopy_detection.src.models.unet import UNet


def load_sample(data_path: str, index: int = 0):
    path = Path(data_path)
    if path.is_dir():
        ds = CHMTileFolderDataset(data_path)
    else:
        ds = CHMDataset(data_path)
    return ds[index]


def main():
    parser = argparse.ArgumentParser(description="Run canopy prediction on one GEDI tile.")
    parser.add_argument("--data", default="data/gedi_tiles", help="Tile .npz file or directory")
    parser.add_argument("--model", default="models/demo_unet.pth", help="Trained U-Net weights")
    parser.add_argument("--index", type=int, default=0, help="Tile index to visualize")
    parser.add_argument("--out", default="visualizations/canopy_prediction.png", help="Output image path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    img, tgt, mask = load_sample(args.data, args.index)

    model = UNet(in_channels=3, out_channels=1, base_filters=16).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).cpu().squeeze(0).squeeze(0)

    tgt = tgt.squeeze(0)
    mask = mask.squeeze(0).bool()
    pred_np = pred.numpy()
    tgt_np = tgt.numpy()
    img_np = img.permute(1, 2, 0).numpy()
    mask_np = mask.numpy()

    masked_pred = np.where(mask_np, pred_np, np.nan)
    masked_tgt = np.where(mask_np, tgt_np, np.nan)
    abs_err = np.abs(masked_pred - masked_tgt)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(np.clip(img_np[:, :, :3], 0, 1))
    axes[0].set_title("Input")
    axes[1].imshow(masked_tgt, cmap="viridis")
    axes[1].set_title("Ground Truth CHM")
    axes[2].imshow(masked_pred, cmap="viridis")
    axes[2].set_title("Predicted CHM")
    axes[3].imshow(abs_err, cmap="magma")
    axes[3].set_title("Absolute Error")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    valid = np.isfinite(abs_err)
    mae = float(np.nanmean(abs_err)) if valid.any() else float("nan")
    print(f"Saved visualization to {out_path}")
    print(f"Masked MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
