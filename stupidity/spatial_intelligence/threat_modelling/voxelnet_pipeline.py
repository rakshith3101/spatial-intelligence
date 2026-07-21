from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .bounding_box_topology import BoundingBoxTopology
from .dataset import SyntheticThreatSceneDataset
from .pygame_scene import PygameThreatWorld
from .scene_dataset import SyntheticVoxelDataset
from .voxel_feature_encoder import VoxelFeatureEncoder
from .voxel_grid_engine import VoxelGridConfig, VoxelGridEngine


class SimpleVoxelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VoxelFeatureEncoder(in_channels=4, hidden_dim=32, out_dim=16)
        self.head = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 7),
        )

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        b, m, t, c = voxels.shape
        features = self.encoder(voxels.reshape(b * m, t, c)).reshape(b, m, -1)
        pooled = features.max(dim=1).values
        return self.head(pooled)


def _load_checkpoint(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    voxelizer = VoxelGridEngine(VoxelGridConfig(**checkpoint["voxel_config"]))
    model = SimpleVoxelNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint, voxelizer, model


def collate_fn(batch, voxelizer: VoxelGridEngine):
    voxel_batches = []
    targets = []
    for item in batch:
        voxels, _coords = voxelizer.encode(item["points"])
        voxel_batches.append(voxels)
        targets.append(item["boxes"][0])
    max_voxels = max(v.shape[0] for v in voxel_batches)
    padded = np.zeros((len(voxel_batches), max_voxels, voxelizer.config.max_points_per_voxel, 4), dtype=np.float32)
    for i, voxels in enumerate(voxel_batches):
        padded[i, : voxels.shape[0]] = voxels
    return torch.from_numpy(padded), torch.from_numpy(np.stack(targets))


def train_and_save(epochs: int = 3, num_samples: int = 1000, output_path: str = "/app/models/voxelnet_threat.pt", device: str = "cpu"):
    voxelizer = VoxelGridEngine(VoxelGridConfig())
    dataset = SyntheticVoxelDataset(num_samples=num_samples)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda batch: collate_fn(batch, voxelizer))

    model = SimpleVoxelNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for voxels, targets in loader:
            voxels = voxels.to(device)
            targets = targets.to(device)
            pred = model(voxels)
            loss = loss_fn(pred[:, :6], targets[:, :6])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        print(f"epoch {epoch + 1}/{epochs} loss={total:.4f}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "voxel_config": asdict(voxelizer.config),
        },
        output,
    )
    print(f"saved checkpoint to {output.resolve()}")
    return output.resolve()


def run_realtime_sim(checkpoint_path: str, steps: int = 120, device: str = "cpu"):
    _, voxelizer, model = _load_checkpoint(checkpoint_path, device)

    dataset = SyntheticVoxelDataset(num_samples=max(steps, 1))
    print("running realtime-style inference loop")
    for step in range(steps):
        sample = dataset[step % len(dataset)]
        voxels, _coords = voxelizer.encode(sample["points"])
        voxels_t = torch.from_numpy(voxels).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(voxels_t).cpu().numpy()[0]
        print(f"step {step + 1:03d}: pred_box={pred[:6]}")


def run_play_mode(
    checkpoint_path: str,
    steps: int = 600,
    device: str = "cpu",
    fps: int = 30,
    record_dir: str | None = None,
):
    if record_dir is not None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    _, voxelizer, model = _load_checkpoint(checkpoint_path, device)
    world = PygameThreatWorld()
    pygame.init()
    headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
    screen = None
    if not headless:
        screen = pygame.display.set_mode(world.arena_size)
        pygame.display.set_caption("Spatial Threat Modeling - Play Mode")
    clock = pygame.time.Clock()
    out_dir = Path(record_dir) if record_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    def steer_away(drone: np.ndarray, threat_center: np.ndarray) -> np.ndarray:
        delta = np.asarray(drone, dtype=np.float32) - np.asarray(threat_center[:2], dtype=np.float32)
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)
        return (delta / norm).astype(np.float32)

    def clamp_velocity(velocity: np.ndarray, max_speed: float = 0.32) -> np.ndarray:
        norm = float(np.linalg.norm(velocity))
        if norm <= max_speed:
            return velocity
        return (velocity / norm * max_speed).astype(np.float32)

    def center_pull(drone: np.ndarray, world_bounds: tuple[float, float, float, float]) -> np.ndarray:
        x_min, x_max, y_min, y_max = world_bounds
        center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=np.float32)
        delta = center - np.asarray(drone, dtype=np.float32)
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (delta / norm).astype(np.float32)

    def edge_repulsion(drone: np.ndarray, world_bounds: tuple[float, float, float, float]) -> np.ndarray:
        x_min, x_max, y_min, y_max = world_bounds
        x, y = map(float, drone)
        force = np.zeros(2, dtype=np.float32)
        margin = 2.0
        if x - x_min < margin:
            force[0] += (margin - (x - x_min)) / margin
        if x_max - x < margin:
            force[0] -= (margin - (x_max - x)) / margin
        if y - y_min < margin:
            force[1] += (margin - (y - y_min)) / margin
        if y_max - y < margin:
            force[1] -= (margin - (y_max - y)) / margin
        norm = float(np.linalg.norm(force))
        if norm < 1e-6:
            return force
        return (force / norm).astype(np.float32)

    running = True
    for step in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if not running:
            break

        scene = world.step()
        points = world.scene_to_point_cloud(scene)
        voxels, _coords = voxelizer.encode(points)
        voxels_t = torch.from_numpy(voxels).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(voxels_t).cpu().numpy()[0]

        predicted_center = pred[:2]
        closest_obj, closest_dist = world.closest_object()

        model_vector = steer_away(world.drone, predicted_center)
        if closest_obj is not None:
            actual_vector = steer_away(world.drone, closest_obj["center"])
            threat_weight = float(np.clip((3.0 - closest_dist) / 3.0, 0.0, 1.0))
        else:
            actual_vector = np.array([0.0, 0.0], dtype=np.float32)
            threat_weight = 0.0

        if closest_dist < 2.8 and closest_obj is not None:
            actual_size = np.asarray(closest_obj["size"][:2], dtype=np.float32)
            escape_speed = 0.12 + 0.22 * threat_weight + 0.05 * float(np.linalg.norm(actual_size))
            desired_velocity = actual_vector * escape_speed
        else:
            desired_velocity = 0.08 * center_pull(world.drone, world.world_bounds)
            desired_velocity += 0.06 * edge_repulsion(world.drone, world.world_bounds)
            desired_velocity += 0.03 * model_vector

        world.drone_velocity = 0.72 * world.drone_velocity + 0.28 * desired_velocity
        world.drone_velocity = clamp_velocity(world.drone_velocity, max_speed=0.18)

        if world.check_collision() and closest_obj is not None:
            world.bounce_drone(closest_obj["center"])
            world.drone_velocity = clamp_velocity(-0.6 * world.drone_velocity + 0.4 * actual_vector)

        if screen is not None:
            world.render(screen)
            pygame.draw.circle(screen, (255, 240, 90), (int(world.arena_size[0] * 0.5), int(world.arena_size[1] * 0.12)), 8)
            pygame.display.flip()

        if out_dir is not None:
            surface = pygame.Surface(world.arena_size)
            world.render(surface)
            pygame.image.save(surface, out_dir / f"frame_{step:04d}.png")

        print(
            f"play step {step + 1:03d}: drone={world.drone}, "
            f"pred_center={predicted_center}, closest_dist={closest_dist:.2f}, threats={sum(o['threat'] for o in scene['objects'])}"
        )
        clock.tick(fps)

    pygame.quit()


def run_pygame_sim(
    steps: int = 300,
    record_dir: str | None = None,
    device: str = "cpu",
    fps: int = 30,
):
    if record_dir is not None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
    pygame.init()
    world = PygameThreatWorld()
    screen = None
    if not headless:
        screen = pygame.display.set_mode(world.arena_size)
        pygame.display.set_caption("Spatial Threat Modeling - Pygame World")
    clock = pygame.time.Clock()
    out_dir = Path(record_dir) if record_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    running = True
    for step in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if not running:
            break

        scene = world.step()
        if screen is not None:
            world.render(screen)
            pygame.display.flip()
        if out_dir is not None:
            surface = pygame.Surface(world.arena_size)
            world.render(surface)
            pygame.image.save(surface, out_dir / f"frame_{step:04d}.png")

        print(f"pygame step {step + 1:03d}: drone={scene['drone']}, threats={sum(o['threat'] for o in scene['objects'])}")
        clock.tick(fps)

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Train and run the synthetic voxel threat pipeline.")
    parser.add_argument("--mode", choices=["train", "infer", "sim", "play"], default="train")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--checkpoint", default="/app/models/voxelnet_threat.pt")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--record-dir", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.mode == "train":
        train_and_save(epochs=args.epochs, num_samples=args.samples, output_path=args.checkpoint, device=args.device)
    elif args.mode == "infer":
        run_realtime_sim(args.checkpoint, steps=args.steps, device=args.device)
    elif args.mode == "play":
        run_play_mode(args.checkpoint, steps=args.steps, device=args.device, fps=args.fps, record_dir=args.record_dir)
    else:
        run_pygame_sim(steps=args.steps, record_dir=args.record_dir, device=args.device, fps=args.fps)


if __name__ == "__main__":
    main()
