from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from canopy_detection.src.models.efficient_unet import EfficientUNet
from canopy_detection.src.phase1_dataset import Phase1CanopyDataset


def masked_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool)
    if not valid.any():
        return float("nan")
    return float(np.abs(pred[valid] - target[valid]).mean())


def masked_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(((pred[valid] - target[valid]) ** 2).mean()))


def masked_bias(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool)
    if not valid.any():
        return float("nan")
    return float((pred[valid] - target[valid]).mean())


def masked_corr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool)
    if valid.sum() < 2:
        return float("nan")
    p = pred[valid].ravel()
    t = target[valid].ravel()
    if np.allclose(p, p[0]) or np.allclose(t, t[0]):
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def save_maps(img: np.ndarray, tgt: np.ndarray, pred: np.ndarray, mask: np.ndarray, out_path: Path, title: str):
    valid = mask.astype(bool)
    err = np.abs(pred - tgt)
    masked_tgt = np.where(valid, tgt, np.nan)
    masked_pred = np.where(valid, pred, np.nan)
    masked_err = np.where(valid, err, np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    rgb = img[:, :, :3] if img.shape[2] >= 3 else img

    axes[0, 0].imshow(np.clip(rgb, 0, 1))
    axes[0, 0].set_title("Input")
    axes[0, 1].imshow(masked_tgt, cmap="viridis")
    axes[0, 1].set_title("Ground Truth CHM")
    axes[0, 2].imshow(masked_pred, cmap="viridis")
    axes[0, 2].set_title("Predicted CHM")
    axes[1, 0].imshow(mask.astype(float), cmap="gray")
    axes[1, 0].set_title("Mask")
    axes[1, 1].imshow(masked_err, cmap="magma")
    axes[1, 1].set_title("Absolute Error")
    axes[1, 2].hist(err[valid].ravel(), bins=30, color="#34495e")
    axes[1, 2].set_title("Error Histogram")
    axes[1, 2].set_xlabel("Absolute error")
    axes[1, 2].set_ylabel("Count")

    for ax in axes.ravel():
        ax.axis("off") if ax is not axes[1, 2] else None

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Inspect phase 1 canopy predictions.")
    parser.add_argument("--data", default="data/phase1_tiles", help="Directory containing phase1 .npz tiles")
    parser.add_argument("--model", default="models/phase1_efficientunet.pth", help="Trained model checkpoint")
    parser.add_argument("--index", type=int, default=0, help="Tile index to inspect")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="visualizations/phase1_prediction.png", help="Output figure path")
    args = parser.parse_args()

    ds = Phase1CanopyDataset(args.data)
    img, tgt, mask = ds[args.index]
    device = torch.device(args.device)

    model = EfficientUNet(in_channels=img.shape[0], out_channels=1, pretrained=False).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).cpu().squeeze(0).squeeze(0).numpy()

    tgt_np = tgt.squeeze(0).numpy()
    mask_np = mask.squeeze(0).numpy().astype(bool)
    img_np = img.permute(1, 2, 0).numpy()

    mae = masked_mae(pred, tgt_np, mask_np)
    rmse = masked_rmse(pred, tgt_np, mask_np)
    bias = masked_bias(pred, tgt_np, mask_np)
    corr = masked_corr(pred, tgt_np, mask_np)

    out_path = Path(args.out)
    save_maps(
        img_np,
        tgt_np,
        pred,
        mask_np,
        out_path,
        title=f"Phase 1 prediction | MAE={mae:.4f} RMSE={rmse:.4f} Bias={bias:.4f} Corr={corr:.4f}",
    )

    print(f"Saved figure to {out_path}")
    print(f"Masked MAE : {mae:.6f}")
    print(f"Masked RMSE: {rmse:.6f}")
    print(f"Masked Bias: {bias:.6f}")
    print(f"Masked Corr: {corr:.6f}")


if __name__ == "__main__":
    main()
