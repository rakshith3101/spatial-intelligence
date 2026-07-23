import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from canopy_detection.src.dataset import CHMDataset, CHMTileFolderDataset
from canopy_detection.src.models.unet import UNet


def masked_l1(pred, target, mask):
    # mask: (B,1,H,W) float
    diff = torch.abs(pred - target) * mask
    denom = mask.sum()
    if denom == 0:
        return diff.mean()
    return diff.sum() / denom


def train(npz_path: str, epochs: int = 5, lr: float = 1e-3, device: str = 'cpu'):
    data_path = Path(npz_path)
    if data_path.is_dir():
        ds = CHMTileFolderDataset(npz_path)
    else:
        ds = CHMDataset(npz_path)
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    model = UNet(in_channels=3, out_channels=1, base_filters=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for img, tgt, mask in dl:
            img = img.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)

            pred = model(img)
            loss = masked_l1(pred, tgt, mask)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        print(f"Epoch {epoch+1}/{epochs} - loss: {total_loss:.6f}")

    torch.save(model.state_dict(), 'models/demo_unet.pth')
    print('Saved model to models/demo_unet.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/demo_chm.npz', help='Path to a single .npz tile or a directory of .npz tiles')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    import os
    os.makedirs('models', exist_ok=True)
    train(args.data, epochs=args.epochs, lr=args.lr, device=args.device)
