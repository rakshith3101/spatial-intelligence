from __future__ import annotations

from pathlib import Path

import numpy as np


def _box_corners(box: np.ndarray) -> np.ndarray:
    x, y, z, w, l, h = box.astype(np.float64)
    dx, dy, dz = w / 2.0, l / 2.0, h / 2.0
    return np.array(
        [
            [x - dx, y - dy, z - dz],
            [x + dx, y - dy, z - dz],
            [x + dx, y + dy, z - dz],
            [x - dx, y + dy, z - dz],
            [x - dx, y - dy, z + dz],
            [x + dx, y - dy, z + dz],
            [x + dx, y + dy, z + dz],
            [x - dx, y + dy, z + dz],
        ],
        dtype=np.float64,
    )


def _draw_box(ax, box: np.ndarray, color: str, label: str) -> None:
    corners = _box_corners(box)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    first = True
    for start, end in edges:
        ax.plot(
            [corners[start, 0], corners[end, 0]],
            [corners[start, 1], corners[end, 1]],
            [corners[start, 2], corners[end, 2]],
            color=color,
            linewidth=2,
            label=label if first else None,
        )
        first = False


def save_spatial_demo_figure(
    points: np.ndarray,
    voxel_coords: np.ndarray,
    voxel_size: tuple[float, float, float],
    spatial_bounds: tuple[float, float, float, float, float, float],
    target_box: np.ndarray,
    pred_box: np.ndarray,
    output_path: str | Path = "visualizations/threat_demo.png",
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=8, alpha=0.35, c=points[:, 3], cmap="viridis", label="points")

    if voxel_coords.size:
        vx, vy, vz = voxel_size
        x_min, x_max, y_min, y_max, z_min, z_max = spatial_bounds
        centers = np.stack(
            [
                x_min + (voxel_coords[:, 0] + 0.5) * vx,
                y_min + (voxel_coords[:, 1] + 0.5) * vy,
                z_min + (voxel_coords[:, 2] + 0.5) * vz,
            ],
            axis=1,
        )
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=40, marker="s", alpha=0.75, c="orange", label="active voxels")

    _draw_box(ax, target_box, "green", "target box")
    _draw_box(ax, pred_box, "red", "predicted box")

    ax.set_title("Spatial Threat Modeling Demo")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left")
    ax.view_init(elev=20, azim=35)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path.resolve()
