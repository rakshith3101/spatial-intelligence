"""
Prepare simple canopy tiles from local GEDI HDF5 granules.

This is a pragmatic starter pipeline for training U-Net on NASA GEDI data.
It searches each HDF5 file for latitude, longitude, and canopy-height-like
metrics, bins them onto a raster grid, and writes .npz tiles that match the
existing canopy_detection dataset format:

  - image: (H, W, C)
  - chm:   (H, W)
  - mask:  (H, W)

Usage:
  python data/prepare_gedi_tiles.py --input-dir data/gedi --out-dir data/gedi_tiles
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


LAT_TOKENS = ("latitude", "lat")
LON_TOKENS = ("longitude", "lon", "long")
TARGET_TOKENS = ("rh100", "elev_highestreturn", "elev_lowestmode", "canopy")
FEATURE_TOKENS = ("rh95", "rh90", "rh50", "elevation", "sensitivity")


def _find_dataset_names(h5_obj):
    names = []

    def _walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            names.append(name)

    h5_obj.visititems(_walk)
    return names


def _find_by_tokens(names, tokens):
    for name in names:
        low = name.lower()
        if any(tok in low for tok in tokens):
            return name
    return None


def _load_arrays(path: Path):
    with h5py.File(path, "r") as f:
        names = _find_dataset_names(f)
        lat_name = _find_by_tokens(names, LAT_TOKENS)
        lon_name = _find_by_tokens(names, LON_TOKENS)
        if not lat_name or not lon_name:
            return None

        target_name = _find_by_tokens(names, TARGET_TOKENS)
        if not target_name:
            return None

        feature_names = []
        for token in FEATURE_TOKENS:
            candidate = _find_by_tokens(names, (token,))
            if candidate and candidate not in feature_names and candidate != target_name:
                feature_names.append(candidate)
            if len(feature_names) == 3:
                break

        lat = np.asarray(f[lat_name][:]).astype(np.float32).ravel()
        lon = np.asarray(f[lon_name][:]).astype(np.float32).ravel()
        target = np.asarray(f[target_name][:]).astype(np.float32).ravel()
        features = [np.asarray(f[name][:]).astype(np.float32).ravel() for name in feature_names]
        return lat, lon, target, features


def _tile_rasterize(lat, lon, target, features, tile_size=128, resolution=None):
    if len(lat) == 0 or len(lon) == 0 or len(target) == 0:
        return None

    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(target)
    for feat in features:
        valid &= np.isfinite(feat)

    lat = lat[valid]
    lon = lon[valid]
    target = target[valid]
    features = [feat[valid] for feat in features]
    if len(lat) == 0:
        return None

    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())
    if resolution is None:
        span = max(lat_max - lat_min, lon_max - lon_min)
        resolution = span / max(tile_size - 1, 1)
        resolution = max(resolution, 1e-6)

    h = tile_size
    w = tile_size
    img = np.zeros((h, w, max(3, len(features))), dtype=np.float32)
    chm = np.full((h, w), np.nan, dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool)
    counts = np.zeros((h, w), dtype=np.int32)

    rows = np.clip(((lat - lat_min) / resolution).astype(np.int32), 0, h - 1)
    cols = np.clip(((lon - lon_min) / resolution).astype(np.int32), 0, w - 1)

    for r, c, t in zip(rows, cols, target):
        if np.isnan(t):
            continue
        if np.isnan(chm[r, c]):
            chm[r, c] = t
        else:
            chm[r, c] = max(chm[r, c], t)
        counts[r, c] += 1
        mask[r, c] = True

    for i, feat in enumerate(features[:3]):
        band = np.zeros((h, w), dtype=np.float32)
        for r, c, v in zip(rows, cols, feat):
            if np.isnan(v):
                continue
            band[r, c] = max(band[r, c], v)
        finite = np.isfinite(band)
        if finite.any():
            band_min = float(np.nanmin(band))
            band_max = float(np.nanmax(band))
            band = (band - band_min) / (band_max - band_min + 1e-8)
        img[:, :, i] = band

    img[:, :, min(3, len(features))] = counts / max(counts.max(), 1)
    img[~mask] = 0.0
    return img, chm, mask


def build_tiles(input_dir: str, out_dir: str, tile_size: int = 128):
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in sorted(input_dir.glob("*.h5")) + sorted(input_dir.glob("*.hdf5")):
        arrays = _load_arrays(path)
        if arrays is None:
            continue
        lat, lon, target, features = arrays
        tile = _tile_rasterize(lat, lon, target, features, tile_size=tile_size)
        if tile is None:
            continue
        img, chm, mask = tile
        out_path = out_dir / f"{path.stem}.npz"
        np.savez_compressed(out_path, image=img, chm=chm, mask=mask)
        written += 1

    print(f"Wrote {written} tile(s) to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/gedi")
    parser.add_argument("--out-dir", default="data/gedi_tiles")
    parser.add_argument("--tile-size", type=int, default=128)
    args = parser.parse_args()
    build_tiles(args.input_dir, args.out_dir, tile_size=args.tile_size)
