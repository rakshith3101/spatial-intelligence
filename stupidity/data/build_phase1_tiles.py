"""
Build phase 1 canopy tiles from aligned Sentinel-2 and GEDI rasters.

This script assumes you already have paired, same-grid arrays for one region:

  - Sentinel tiles: image inputs with 4 or more bands
  - GEDI tiles: canopy height labels and a valid-data mask

Input conventions:
  - Sentinel tiles are .npz files containing:
      * image: (H, W, C)
      * optional mask: (H, W)
  - GEDI tiles are .npz files containing:
      * chm: (H, W)
      * mask: (H, W)

The pairing is by shared stem name:
  sentinel/india_001.npz  <->  gedi/india_001.npz

Output:
  - image: (H, W, C + extra_channels)
  - chm: (H, W)
  - mask: (H, W)

Usage:
  python data/build_phase1_tiles.py --sentinel-dir data/sentinel_tiles --gedi-dir data/gedi_tiles --out-dir data/phase1_tiles
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CHM_MAX = 60.0

def _load_npz(path: Path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _normalize_bandstack(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[2]):
        band = img[:, :, i]
        finite = np.isfinite(band)
        if not finite.any():
            continue
        band_min = float(np.nanmin(band))
        band_max = float(np.nanmax(band))
        out[:, :, i] = (band - band_min) / (band_max - band_min + 1e-8)
    return np.nan_to_num(out, 0.0)


def _edge_channel(mask: np.ndarray) -> np.ndarray:
    mask_f = mask.astype(np.float32)
    grad_y = np.abs(np.diff(mask_f, axis=0, prepend=mask_f[:1, :]))
    grad_x = np.abs(np.diff(mask_f, axis=1, prepend=mask_f[:, :1]))
    return np.clip(grad_x + grad_y, 0.0, 1.0)


def build_phase1_tiles(sentinel_dir: str, gedi_dir: str, out_dir: str):
    sentinel_dir = Path(sentinel_dir)
    gedi_dir = Path(gedi_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentinel_files = {p.stem: p for p in sentinel_dir.glob("*.npz")}
    gedi_files = {p.stem: p for p in gedi_dir.glob("*.npz")}
    common = sorted(set(sentinel_files) & set(gedi_files))

    if not common:
        raise FileNotFoundError("No matching Sentinel/GEDI tile stems found.")

    written = 0
    for stem in common:
        s = _load_npz(sentinel_files[stem])
        g = _load_npz(gedi_files[stem])

        if "image" not in s or "chm" not in g or "mask" not in g:
            continue

        sentinel_img = s["image"].astype(np.float32)
        gedi_chm = g["chm"].astype(np.float32)
        gedi_mask = g["mask"].astype(bool)
        sentinel_mask = s.get("mask", np.ones(gedi_mask.shape, dtype=bool)).astype(bool)

        if sentinel_img.shape[:2] != gedi_chm.shape[:2]:
            raise ValueError(
                f"Tile size mismatch for {stem}: "
                f"sentinel={sentinel_img.shape[:2]} gedi={gedi_chm.shape[:2]}"
            )

        sentinel_img = _normalize_bandstack(sentinel_img)
        valid_mask = sentinel_mask & gedi_mask & np.isfinite(gedi_chm)

        if not valid_mask.any():
            continue

        extra = [
            valid_mask.astype(np.float32)[..., None],
            _edge_channel(valid_mask)[..., None],
        ]
        image = np.concatenate([sentinel_img, *extra], axis=2)

        out_path = out_dir / f"{stem}.npz"
        chm = np.clip(np.nan_to_num(gedi_chm, 0.0), 0.0, CHM_MAX)
        np.savez_compressed(out_path, image=image, chm=chm, mask=valid_mask)
        written += 1

    print(f"Wrote {written} phase 1 tile(s) to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build phase 1 canopy tiles from aligned inputs.")
    parser.add_argument("--sentinel-dir", default="data/sentinel_tiles")
    parser.add_argument("--gedi-dir", default="data/gedi_tiles")
    parser.add_argument("--out-dir", default="data/phase1_tiles")
    args = parser.parse_args()
    build_phase1_tiles(args.sentinel_dir, args.gedi_dir, args.out_dir)
