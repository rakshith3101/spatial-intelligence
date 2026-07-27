"""
Convert a Sentinel-2 .SAFE product into a simplified training tile.

This is a lightweight preprocessing step that:
  - finds 10m band JP2 files inside the SAFE archive
  - reads them with OpenCV
  - resizes/crops them to a common square tile
  - stacks the bands into an image tensor saved as .npz

The output format matches the phase 1 tile builder:
  - image: (H, W, C)
  - mask: (H, W)

This script is intentionally pragmatic and avoids requiring rasterio/GDAL.
It assumes you already picked a Sentinel scene that overlaps your GEDI area.

Usage:
  python data/prepare_sentinel2_tiles.py --safe-dir "path\\to\\S2...SAFE" --out-dir data/sentinel_tiles --stem india_001
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_BANDS_10M = ("B02", "B03", "B04", "B08")


def _find_band_files(safe_dir: Path, bands: tuple[str, ...]) -> dict[str, Path]:
    band_files: dict[str, Path] = {}
    for jp2 in safe_dir.rglob("*.jp2"):
        path_str = str(jp2).upper()
        name = jp2.name.upper()

        # Skip quality and metadata JP2s; keep only image data bands.
        if "QI_DATA" in path_str or "AUX_DATA" in path_str:
            continue
        if "IMG_DATA" not in path_str:
            continue

        for band in bands:
            if f"_{band}_" in name or f"_{band}." in name or f"_{band}_" in name:
                band_files[band] = jp2
    return band_files


def _read_jp2(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Could not read band file: {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr.astype(np.float32)


def _normalize_band(band: np.ndarray) -> np.ndarray:
    finite = np.isfinite(band)
    if not finite.any():
        return np.zeros_like(band, dtype=np.float32)
    bmin = float(np.nanmin(band))
    bmax = float(np.nanmax(band))
    return ((band - bmin) / (bmax - bmin + 1e-8)).astype(np.float32)


def _center_crop_or_pad(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == size and w == size:
        return img

    out = np.zeros((size, size) + (() if img.ndim == 2 else (img.shape[2],)), dtype=img.dtype)
    y0 = max((size - h) // 2, 0)
    x0 = max((size - w) // 2, 0)
    sy0 = max((h - size) // 2, 0)
    sx0 = max((w - size) // 2, 0)
    hh = min(h, size)
    ww = min(w, size)
    out[y0 : y0 + hh, x0 : x0 + ww] = img[sy0 : sy0 + hh, sx0 : sx0 + ww]
    return out


def prepare_sentinel_tile(safe_dir: str, out_dir: str, stem: str, tile_size: int = 128, bands: tuple[str, ...] = DEFAULT_BANDS_10M):
    safe_dir = Path(safe_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if safe_dir.is_file():
        raise NotADirectoryError(f"Expected a .SAFE directory, got file: {safe_dir}")
    if not safe_dir.exists():
        raise FileNotFoundError(f"Sentinel SAFE directory not found: {safe_dir}")

    # Allow callers to pass a SAFE root or a nested GRANULE folder.
    granule_dir = None
    if (safe_dir / "GRANULE").exists():
        granule_dir = safe_dir
    else:
        granule_candidates = list(safe_dir.rglob("GRANULE"))
        if granule_candidates:
            granule_dir = granule_candidates[0].parent
    if granule_dir is not None:
        safe_dir = granule_dir

    band_files = _find_band_files(safe_dir, bands)
    missing = [b for b in bands if b not in band_files]
    if missing:
        raise FileNotFoundError(f"Missing Sentinel-2 bands in SAFE: {missing}")

    stacked = []
    for band in bands:
        arr = _read_jp2(band_files[band])
        arr = _center_crop_or_pad(arr, tile_size)
        stacked.append(_normalize_band(arr))

    image = np.stack(stacked, axis=-1).astype(np.float32)
    mask = np.isfinite(image).all(axis=2)
    image[~mask] = 0.0

    out_path = out_dir / f"{stem}.npz"
    np.savez_compressed(out_path, image=image, mask=mask)
    print(f"Saved Sentinel tile to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Sentinel-2 SAFE to a simplified training tile.")
    parser.add_argument("--safe-dir", required=True, help="Path to the .SAFE directory")
    parser.add_argument("--out-dir", default="data/sentinel_tiles", help="Output directory for .npz tile")
    parser.add_argument("--stem", required=True, help="Output tile stem, e.g. india_001")
    parser.add_argument("--tile-size", type=int, default=128)
    args = parser.parse_args()
    prepare_sentinel_tile(args.safe_dir, args.out_dir, args.stem, tile_size=args.tile_size)
