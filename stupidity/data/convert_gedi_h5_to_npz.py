"""
Convert GEDI L2A HDF5 granules into simplified raster tiles for canopy training.

This script is intentionally lightweight:
  - scans each .h5/.hdf5 file for latitude, longitude, and a canopy-height-like target
  - rasterizes the footprint samples into a fixed square tile
  - saves a .npz with image/chm/mask arrays

Output format:
  - image: (H, W, C)   feature stack
  - chm:   (H, W)      canopy-height target
  - mask:  (H, W)      valid-data mask

Typical usage:
  python data/convert_gedi_h5_to_npz.py --input-dir data/gedi --out-dir data/gedi_tiles
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
CHM_MIN = 0.0
CHM_MAX = 60.0


def _dataset_names(h5_obj):
    names = []

    def _walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            names.append(name)

    h5_obj.visititems(_walk)
    return names


def _find_name(names, tokens):
    for name in names:
        low = name.lower()
        if any(tok in low for tok in tokens):
            return name
    return None


def _load_h5(path: Path):
    with h5py.File(path, "r") as f:
        names = _dataset_names(f)
        lat_name = _find_name(names, LAT_TOKENS)
        lon_name = _find_name(names, LON_TOKENS)
        target_name = _find_name(names, TARGET_TOKENS)
        if not lat_name or not lon_name or not target_name:
            return None

        feature_names = []
        for token in FEATURE_TOKENS:
            name = _find_name(names, (token,))
            if name and name not in feature_names and name != target_name:
                feature_names.append(name)
            if len(feature_names) >= 3:
                break

        lat = np.asarray(f[lat_name][:], dtype=np.float32).ravel()
        lon = np.asarray(f[lon_name][:], dtype=np.float32).ravel()
        target = np.asarray(f[target_name][:], dtype=np.float32).ravel()
        features = [np.asarray(f[name][:], dtype=np.float32).ravel() for name in feature_names]
        return lat, lon, target, features


def _normalize(a: np.ndarray) -> np.ndarray:
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=np.float32)
    amin = float(np.nanmin(a))
    amax = float(np.nanmax(a))
    return ((a - amin) / (amax - amin + 1e-8)).astype(np.float32)


def _rasterize(lat, lon, target, features, tile_size: int):
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
    span = max(lat_max - lat_min, lon_max - lon_min)
    resolution = max(span / max(tile_size - 1, 1), 1e-6)

    rows = np.clip(((lat - lat_min) / resolution).astype(np.int32), 0, tile_size - 1)
    cols = np.clip(((lon - lon_min) / resolution).astype(np.int32), 0, tile_size - 1)

    chm = np.full((tile_size, tile_size), np.nan, dtype=np.float32)
    mask = np.zeros((tile_size, tile_size), dtype=bool)
    counts = np.zeros((tile_size, tile_size), dtype=np.int32)

    for r, c, t in zip(rows, cols, target):
        if np.isnan(t):
            continue
        t = float(np.clip(t, CHM_MIN, CHM_MAX))
        if np.isnan(chm[r, c]):
            chm[r, c] = t
        else:
            chm[r, c] = max(chm[r, c], t)
        mask[r, c] = True
        counts[r, c] += 1

    bands = []
    for feat in features[:3]:
        band = np.zeros((tile_size, tile_size), dtype=np.float32)
        for r, c, v in zip(rows, cols, feat):
            if np.isnan(v):
                continue
            band[r, c] = max(band[r, c], v)
        bands.append(_normalize(band))

    bands.append(_normalize(counts.astype(np.float32)))
    image = np.stack(bands, axis=-1).astype(np.float32)
    image[~mask] = 0.0
    chm = np.clip(np.nan_to_num(chm, 0.0), CHM_MIN, CHM_MAX).astype(np.float32)
    return image, chm, mask


def convert_dir(input_dir: str, out_dir: str, tile_size: int = 128):
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in sorted(input_dir.glob("*.h5")) + sorted(input_dir.glob("*.hdf5")):
        arrays = _load_h5(path)
        if arrays is None:
            print(f"Skipping {path.name}: could not find GEDI lat/lon/target fields")
            continue
        lat, lon, target, features = arrays
        tile = _rasterize(lat, lon, target, features, tile_size=tile_size)
        if tile is None:
            continue
        image, chm, mask = tile
        out_path = out_dir / f"{path.stem}.npz"
        np.savez_compressed(out_path, image=image, chm=chm, mask=mask)
        written += 1

    print(f"Wrote {written} GEDI tile(s) to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GEDI HDF5 granules to raster tiles.")
    parser.add_argument("--input-dir", default="data/gedi", help="Folder containing GEDI .h5 files")
    parser.add_argument("--out-dir", default="data/gedi_tiles", help="Folder for output .npz tiles")
    parser.add_argument("--tile-size", type=int, default=128, help="Output tile size in pixels")
    args = parser.parse_args()
    convert_dir(args.input_dir, args.out_dir, tile_size=args.tile_size)
