from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from canopy_detection.src.phase1_dataset import Phase1CanopyDataset
from canopy_detection.src.models.efficient_unet import EfficientUNet


def masked_l1(pred, target, mask):
    diff = torch.abs(pred - target) * mask
    denom = mask.sum()
    if denom == 0:
        return diff.mean()
    return diff.sum() / denom


def masked_rmse(pred, target, mask):
    diff2 = ((pred - target) ** 2) * mask
    denom = mask.sum()
    if denom == 0:
        return torch.sqrt(diff2.mean())
    return torch.sqrt(diff2.sum() / denom)


def masked_mae(pred, target, mask):
    return masked_l1(pred, target, mask)


def _evaluate(model, dl, device):
    model.eval()
    totals = {"loss": 0.0, "mae": 0.0, "rmse": 0.0}
    n = 0
    with torch.no_grad():
        for img, tgt, mask in dl:
            img = img.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)
            pred = model(img)
            totals["loss"] += float(masked_l1(pred, tgt, mask).item())
            totals["mae"] += float(masked_mae(pred, tgt, mask).item())
            totals["rmse"] += float(masked_rmse(pred, tgt, mask).item())
            n += 1
    if n == 0:
        return {k: float("nan") for k in totals}
    return {k: v / n for k, v in totals.items()}


def train(data_dir: str, epochs: int = 10, lr: float = 1e-4, device: str = "cpu", val_split: float = 0.2):
    ds = Phase1CanopyDataset(data_dir)
    if len(ds) > 1 and 0.0 < val_split < 1.0:
        val_len = max(1, int(len(ds) * val_split))
        train_len = len(ds) - val_len
        train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    else:
        train_ds, val_ds = ds, None

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1) if val_ds is not None else None

    sample_img, _, _ = ds[0]
    in_channels = sample_img.shape[0]
    model = EfficientUNet(in_channels=in_channels, out_channels=1, pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    best_val = float("inf")
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    history = []

    run_config = {
        "data_dir": str(data_dir),
        "epochs": epochs,
        "lr": lr,
        "device": device,
        "val_split": val_split,
        "in_channels": int(in_channels),
        "tile_count": len(ds),
    }
    (out_dir / "phase1_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_steps = 0
        for img, tgt, mask in train_dl:
            img = img.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)

            pred = model(img)
            loss = masked_l1(pred, tgt, mask)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            total_steps += 1

        train_loss = total_loss / max(total_steps, 1)
        msg = f"Epoch {epoch + 1}/{epochs} - train loss: {train_loss:.6f}"

        val_metrics = None
        if val_dl is not None:
            val_metrics = _evaluate(model, val_dl, device)
            msg += (
                f" - val loss: {val_metrics['loss']:.6f}"
                f" - val mae: {val_metrics['mae']:.6f}"
                f" - val rmse: {val_metrics['rmse']:.6f}"
            )
            sched.step(val_metrics["loss"])
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(model.state_dict(), out_dir / "phase1_efficientunet.pth")
        else:
            sched.step(train_loss)
        print(msg)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": None if val_metrics is None else val_metrics["loss"],
                "val_mae": None if val_metrics is None else val_metrics["mae"],
                "val_rmse": None if val_metrics is None else val_metrics["rmse"],
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )

    if val_dl is None:
        torch.save(model.state_dict(), out_dir / "phase1_efficientunet.pth")

    (out_dir / "phase1_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved model to {out_dir / 'phase1_efficientunet.pth'}")
    print(f"Saved training history to {out_dir / 'phase1_history.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train phase 1 canopy height model.")
    parser.add_argument("--data", default="data/phase1_tiles", help="Directory of aligned .npz tiles")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()
    train(args.data, epochs=args.epochs, lr=args.lr, device=args.device, val_split=args.val_split)
