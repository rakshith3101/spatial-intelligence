import os
import argparse
import numpy as np
from spatial_intelligence.LiDAR import LidarSpatialEngine


def rasterize_point_cloud(pcd_points: np.ndarray, resolution: float = 0.5):
    """Rasterize point cloud to DSM (max Z) and DTM (min Z) grids.

    Args:
        pcd_points: (N,3) array of x,y,z
        resolution: grid cell size in the same units as points

    Returns:
        dsm, dtm, chm, mask, x_edges, y_edges
    """
    xs = pcd_points[:, 0]
    ys = pcd_points[:, 1]
    zs = pcd_points[:, 2]

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    nx = int(np.ceil((xmax - xmin) / resolution)) + 1
    ny = int(np.ceil((ymax - ymin) / resolution)) + 1

    # cell indices
    ix = np.floor((xs - xmin) / resolution).astype(np.int64)
    iy = np.floor((ys - ymin) / resolution).astype(np.int64)

    flat_idx = iy * nx + ix

    size = nx * ny
    dsm_flat = np.full(size, -np.inf, dtype=float)
    dtm_flat = np.full(size, np.inf, dtype=float)

    # aggregate using numpy at-ops
    np.maximum.at(dsm_flat, flat_idx, zs)
    np.minimum.at(dtm_flat, flat_idx, zs)

    dsm = dsm_flat.reshape((ny, nx))
    dtm = dtm_flat.reshape((ny, nx))

    # mask where we have any points
    valid = np.isfinite(dsm) & np.isfinite(dtm)

    dsm[~valid] = np.nan
    dtm[~valid] = np.nan

    chm = dsm - dtm
    chm[~valid] = np.nan

    x_edges = xmin + np.arange(nx + 1) * resolution
    y_edges = ymin + np.arange(ny + 1) * resolution

    return dsm, dtm, chm, valid, x_edges, y_edges


def build_synthetic_image(dsm: np.ndarray, dtm: np.ndarray, mask: np.ndarray):
    """Create a 3-channel synthetic 'image' from DSM/DTM for demo training.

    Channels:
      0: normalized DSM (0..1)
      1: normalized DTM (0..1)
      2: normalized local slope / gradient magnitude
    """
    import scipy.ndimage as ndi

    # simple normalization using finite values
    dsm_f = np.copy(dsm)
    dtm_f = np.copy(dtm)
    finite = np.isfinite(dsm_f)
    if finite.sum() == 0:
        raise RuntimeError("Empty DSM - no points to rasterize")

    dsm_min, dsm_max = np.nanmin(dsm_f), np.nanmax(dsm_f)
    dtm_min, dtm_max = np.nanmin(dtm_f), np.nanmax(dtm_f)

    dsm_n = (dsm_f - dsm_min) / (dsm_max - dsm_min + 1e-8)
    dtm_n = (dtm_f - dtm_min) / (dtm_max - dtm_min + 1e-8)

    # gradient magnitude as proxy texture
    grad_x = ndi.sobel(np.nan_to_num(dsm_n, 0.0), axis=1)
    grad_y = ndi.sobel(np.nan_to_num(dsm_n, 0.0), axis=0)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    gm_n = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)

    # stack and set zeros where mask is false
    h, w = dsm_n.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, :, 0] = np.nan_to_num(dsm_n, 0.0)
    img[:, :, 1] = np.nan_to_num(dtm_n, 0.0)
    img[:, :, 2] = np.nan_to_num(gm_n, 0.0)

    img[~mask] = 0.0

    return img


def main(out_path: str = "data/demo_chm.npz", resolution: float = 0.5):
    engine = LidarSpatialEngine()
    pcd = engine.generate_synthetic_point_cloud()
    points = np.asarray(pcd.points)

    dsm, dtm, chm, mask, x_edges, y_edges = rasterize_point_cloud(points, resolution=resolution)

    img = build_synthetic_image(dsm, dtm, mask)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, image=img, chm=chm.astype(np.float32), mask=mask)
    print(f"Saved demo CHM data to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", "-o", default="data/demo_chm.npz")
    parser.add_argument("--res", type=float, default=0.5)
    args = parser.parse_args()
    main(args.out, resolution=args.res)
