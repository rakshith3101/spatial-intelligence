from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .bounding_box_topology import BoundingBoxTopology
from .dataset import SyntheticThreatSceneDataset
from .tensor_projection_node import TensorProjectionNode
from .visualize import save_spatial_demo_figure
from .voxel_feature_encoder import VoxelFeatureEncoder
from .voxel_grid_engine import VoxelGridConfig, VoxelGridEngine

class ThreatBoxHead(nn.Module):
    def __init__(self, in_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6),
        )

    def forward(self, x):
        return self.net(x)


def collate_batch(batch, voxelizer: VoxelGridEngine, projector: TensorProjectionNode, pose_matrix: np.ndarray):
    voxel_batches = []
    targets = []
    for cloud, target_boxes, _class_ids, _point_labels, _metadata in batch:
        world_points = projector.transform_point_cloud_tensor(cloud[:, :3], pose_matrix)
        world_cloud = np.hstack([world_points, cloud[:, 3:4]])
        voxels, _coords = voxelizer.encode(world_cloud)
        voxel_batches.append(voxels)
        targets.append(target_boxes[0, :6])

    max_voxels = max(v.shape[0] for v in voxel_batches)
    padded = np.zeros((len(voxel_batches), max_voxels, voxelizer.config.max_points_per_voxel, 4), dtype=np.float32)
    for i, voxels in enumerate(voxel_batches):
        padded[i, : voxels.shape[0]] = voxels
    return torch.from_numpy(padded), torch.from_numpy(np.stack(targets))


def run_demo(epochs: int = 3, device: str = "cpu") -> None:
    projector = TensorProjectionNode()
    voxelizer = VoxelGridEngine(VoxelGridConfig())
    pose = projector.construct_se3_matrix(roll_deg=0.0, pitch_deg=8.0, yaw_deg=20.0, translation=np.array([1.5, -0.5, 0.2]))
    dataset = SyntheticThreatSceneDataset(num_scenes=64, seed=7)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda batch: collate_batch(batch, voxelizer, projector, pose))

    encoder = VoxelFeatureEncoder().to(device)
    head = ThreatBoxHead().to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)
    criterion = nn.SmoothL1Loss()
    bbox_eval = BoundingBoxTopology()

    for epoch in range(epochs):
        encoder.train()
        head.train()
        total = 0.0
        for voxels, targets in loader:
            voxels = voxels.to(device)
            targets = targets.to(device)
            b, m, t, c = voxels.shape
            features = encoder(voxels.reshape(b * m, t, c)).reshape(b, m, -1)
            pooled = features.max(dim=1).values
            pred = head(pooled)
            loss = criterion(pred, targets[:, :6])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        print(f"epoch {epoch + 1}/{epochs} loss={total:.4f}")

    encoder.eval()
    head.eval()
    sample_cloud, sample_boxes, sample_class_ids, sample_point_labels, sample_metadata = dataset[0]
    world_points = projector.transform_point_cloud_tensor(sample_cloud[:, :3], pose)
    world_cloud = np.hstack([world_points, sample_cloud[:, 3:4]])
    voxels, coords = voxelizer.encode(world_cloud)
    voxels_t = torch.from_numpy(voxels).unsqueeze(0).to(device)
    features = encoder(voxels_t.reshape(voxels_t.shape[0] * voxels_t.shape[1], voxels_t.shape[2], voxels_t.shape[3])).reshape(1, voxels_t.shape[1], -1)
    pooled = features.max(dim=1).values
    pred_box = head(pooled).detach().cpu().numpy()[0]
    target_box = sample_boxes[0, :6]
    iou = bbox_eval.calculate_axis_aligned_3d_iou(target_box, pred_box)
    fig_path = save_spatial_demo_figure(
        points=world_cloud,
        voxel_coords=coords,
        voxel_size=voxelizer.config.voxel_size,
        spatial_bounds=voxelizer.config.spatial_bounds,
        target_box=target_box,
        pred_box=pred_box,
    )

    print("sample point cloud:", sample_cloud.shape)
    print("transformed points:", world_points.shape)
    print("voxel tensor:", voxels.shape, "active voxels:", coords.shape[0])
    print("encoded features:", tuple(features.shape))
    print("predicted box:", pred_box)
    print("target box:", target_box)
    print("scene metadata:", sample_metadata)
    print("iou:", round(iou, 4))
    print("visualization saved to:", fig_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the spatial threat modeling demo.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_demo(epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()
