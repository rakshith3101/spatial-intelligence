from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np


@dataclass
class SimObject:
    class_id: int
    center: np.ndarray
    size: np.ndarray
    velocity: np.ndarray

    def step(self, bounds: tuple[float, float, float, float]) -> None:
        self.center[:2] += self.velocity[:2]
        x_min, x_max, y_min, y_max = bounds
        half = self.size[:2] / 2.0
        if self.center[0] - half[0] < x_min or self.center[0] + half[0] > x_max:
            self.velocity[0] *= -1
            self.center[0] = np.clip(self.center[0], x_min + half[0], x_max - half[0])
        if self.center[1] - half[1] < y_min or self.center[1] + half[1] > y_max:
            self.velocity[1] *= -1
            self.center[1] = np.clip(self.center[1], y_min + half[1], y_max - half[1])


class PygameThreatWorld:
    """Top-down 2D world that generates synthetic 3D threat scenes."""

    def __init__(
        self,
        seed: int = 7,
        arena_size: tuple[int, int] = (900, 700),
        world_bounds: tuple[float, float, float, float] = (-10.0, 10.0, -8.0, 8.0),
        num_objects: int = 5,
    ):
        self.rng = np.random.default_rng(seed)
        self.arena_size = arena_size
        self.world_bounds = world_bounds
        self.objects: list[SimObject] = []
        self.drone = np.array([0.0, 0.0], dtype=np.float32)
        self.drone_velocity = np.array([0.03, 0.02], dtype=np.float32)
        self._spawn_objects(num_objects)

    def _spawn_objects(self, num_objects: int) -> None:
        x_min, x_max, y_min, y_max = self.world_bounds
        near_drone_spots = [
            (-1.5, 0.8),
            (1.2, 1.6),
            (2.0, -0.8),
            (-2.2, -1.2),
        ]
        for i in range(num_objects):
            if i < len(near_drone_spots):
                base_x, base_y = near_drone_spots[i]
                center_xy = np.array(
                    [
                        np.clip(base_x + self.rng.normal(0.0, 0.6), x_min + 1, x_max - 1),
                        np.clip(base_y + self.rng.normal(0.0, 0.6), y_min + 1, y_max - 1),
                    ],
                    dtype=np.float32,
                )
            else:
                center_xy = np.array(
                    [self.rng.uniform(x_min + 1, x_max - 1), self.rng.uniform(y_min + 1, y_max - 1)],
                    dtype=np.float32,
                )
            center = np.array(
                [center_xy[0], center_xy[1], self.rng.uniform(0.0, 1.0)],
                dtype=np.float32,
            )
            size = np.array(
                [self.rng.uniform(0.8, 2.5), self.rng.uniform(0.8, 2.0), self.rng.uniform(0.5, 2.2)],
                dtype=np.float32,
            )
            velocity = np.array([self.rng.uniform(-0.08, 0.08), self.rng.uniform(-0.08, 0.08), 0.0], dtype=np.float32)
            self.objects.append(SimObject(class_id=(i % 3) + 1, center=center, size=size, velocity=velocity))

    def step(self) -> dict:
        x_min, x_max, y_min, y_max = self.world_bounds
        self.drone += self.drone_velocity
        if self.drone[0] < x_min or self.drone[0] > x_max:
            self.drone_velocity[0] *= -1
            self.drone[0] = np.clip(self.drone[0], x_min, x_max)
        if self.drone[1] < y_min or self.drone[1] > y_max:
            self.drone_velocity[1] *= -1
            self.drone[1] = np.clip(self.drone[1], y_min, y_max)

        for obj in self.objects:
            obj.step(self.world_bounds)

        return self.export_scene()

    def export_scene(self) -> dict:
        threat_threshold = 3.0
        objects = []
        for obj in self.objects:
            dx = obj.center[0] - self.drone[0]
            dy = obj.center[1] - self.drone[1]
            dist = math.sqrt(dx * dx + dy * dy)
            objects.append(
                {
                    "class_id": obj.class_id,
                    "center": obj.center.copy(),
                    "size": obj.size.copy(),
                    "velocity": obj.velocity.copy(),
                    "threat": float(dist < threat_threshold),
                }
            )

        return {
            "drone": self.drone.copy(),
            "objects": objects,
            "world_bounds": self.world_bounds,
            "arena_size": self.arena_size,
        }

    def closest_object(self) -> tuple[dict | None, float]:
        closest = None
        closest_dist = float("inf")
        for obj in self.objects:
            delta = obj.center[:2] - self.drone
            dist = float(np.linalg.norm(delta))
            if dist < closest_dist:
                closest = obj
                closest_dist = dist
        if closest is None:
            return None, float("inf")
        return {
            "class_id": closest.class_id,
            "center": closest.center.copy(),
            "size": closest.size.copy(),
            "velocity": closest.velocity.copy(),
        }, closest_dist

    def check_collision(self) -> bool:
        for obj in self.objects:
            half = obj.size[:2] / 2.0
            if (
                abs(self.drone[0] - obj.center[0]) <= half[0]
                and abs(self.drone[1] - obj.center[1]) <= half[1]
            ):
                return True
        return False

    def bounce_drone(self, away_from: np.ndarray) -> None:
        delta = self.drone - np.asarray(away_from[:2], dtype=np.float32)
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            delta = np.array([1.0, 0.0], dtype=np.float32)
            norm = 1.0
        direction = delta / norm
        self.drone += direction * 0.75
        self.drone[0] = np.clip(self.drone[0], self.world_bounds[0], self.world_bounds[1])
        self.drone[1] = np.clip(self.drone[1], self.world_bounds[2], self.world_bounds[3])

    def scene_to_point_cloud(self, scene: dict, points_per_object: int = 96, background_points: int = 192) -> np.ndarray:
        """Convert the current 2D scene into a synthetic 3D point cloud."""
        points = []
        for obj in scene["objects"]:
            center = np.asarray(obj["center"], dtype=np.float32)
            size = np.asarray(obj["size"], dtype=np.float32)
            spread = np.maximum(size / 4.0, 0.05)
            xyz = center + self.rng.normal(scale=spread, size=(points_per_object, 3)).astype(np.float32)
            intensity = np.full((points_per_object, 1), 0.75 if obj["threat"] else 0.35, dtype=np.float32)
            points.append(np.hstack([xyz, intensity]))

        x_min, x_max, y_min, y_max = self.world_bounds
        bg_xyz = np.stack(
            [
                self.rng.uniform(x_min, x_max, size=background_points),
                self.rng.uniform(y_min, y_max, size=background_points),
                self.rng.uniform(-0.3, 1.2, size=background_points),
            ],
            axis=1,
        ).astype(np.float32)
        bg_i = self.rng.uniform(0.0, 0.2, size=(background_points, 1)).astype(np.float32)
        points.append(np.hstack([bg_xyz, bg_i]))
        return np.concatenate(points, axis=0).astype(np.float32)

    def render(self, screen) -> None:
        import pygame

        screen.fill((12, 14, 22))
        width, height = self.arena_size
        x_min, x_max, y_min, y_max = self.world_bounds

        def map_point(point: np.ndarray) -> tuple[int, int]:
            x = int((point[0] - x_min) / (x_max - x_min) * width)
            y = int(height - ((point[1] - y_min) / (y_max - y_min) * height))
            return x, y

        for obj in self.objects:
            x, y = map_point(obj.center)
            box_w = int(obj.size[0] / (x_max - x_min) * width)
            box_h = int(obj.size[1] / (y_max - y_min) * height)
            color = (220, 90, 90) if obj.class_id == 1 else (90, 180, 220) if obj.class_id == 2 else (220, 180, 80)
            rect = pygame.Rect(0, 0, max(12, box_w), max(12, box_h))
            rect.center = (x, y)
            pygame.draw.rect(screen, color, rect, border_radius=6)

        drone_pos = map_point(self.drone)
        pygame.draw.circle(screen, (120, 230, 140), drone_pos, 12)
        pygame.draw.circle(screen, (120, 230, 140), drone_pos, 80, 1)
        pygame.draw.circle(screen, (255, 240, 90), drone_pos, 5)
