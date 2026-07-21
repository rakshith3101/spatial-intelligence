from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ThreatObjectSpec:
    class_name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw_deg: float
    intensity: float


@dataclass(frozen=True)
class SyntheticScene:
    scene_id: str
    points: np.ndarray
    point_labels: np.ndarray
    boxes: np.ndarray
    class_ids: np.ndarray
    metadata: dict


class SyntheticSceneGenerator:
    """Generate labeled 3D scenes for phase-1 detector learning."""

    def __init__(
        self,
        seed: int = 7,
        roi_bounds: tuple[float, float, float, float, float, float] = (-10.0, 10.0, -10.0, 10.0, -3.0, 3.0),
        num_background_points: int = 256,
        points_per_object: tuple[int, int] = (96, 192),
    ):
        self.rng = np.random.default_rng(seed)
        self.roi_bounds = roi_bounds
        self.num_background_points = num_background_points
        self.points_per_object = points_per_object

        self.class_palette = {
            "background": 0,
            "vehicle": 1,
            "person": 2,
            "drone": 3,
            "animal": 4,
            "unknown": 5,
        }

    def _sample_object(self) -> ThreatObjectSpec:
        class_name = self.rng.choice(["vehicle", "person", "drone", "animal", "unknown"])
        if class_name == "vehicle":
            size = self.rng.uniform([1.4, 2.4, 1.2], [2.5, 4.5, 2.2])
            center = self.rng.uniform([-6.0, -6.0, -0.2], [6.0, 6.0, 1.0])
        elif class_name == "person":
            size = self.rng.uniform([0.4, 0.4, 1.4], [0.8, 0.8, 2.0])
            center = self.rng.uniform([-7.0, -7.0, -0.1], [7.0, 7.0, 1.8])
        elif class_name == "drone":
            size = self.rng.uniform([0.2, 0.2, 0.1], [0.8, 0.8, 0.5])
            center = self.rng.uniform([-5.0, -5.0, 0.5], [5.0, 5.0, 3.0])
        elif class_name == "animal":
            size = self.rng.uniform([0.3, 0.3, 0.2], [1.0, 1.5, 1.0])
            center = self.rng.uniform([-8.0, -8.0, -0.1], [8.0, 8.0, 1.0])
        else:
            size = self.rng.uniform([0.2, 0.2, 0.2], [2.0, 2.0, 2.0])
            center = self.rng.uniform([-8.0, -8.0, -0.5], [8.0, 8.0, 2.0])

        yaw = float(self.rng.uniform(-180.0, 180.0))
        intensity = float(self.rng.uniform(0.2, 1.0))
        return ThreatObjectSpec(class_name, tuple(center.tolist()), tuple(size.tolist()), yaw, intensity)

    def _object_points(self, spec: ThreatObjectSpec) -> np.ndarray:
        count = int(self.rng.integers(self.points_per_object[0], self.points_per_object[1] + 1))
        center = np.asarray(spec.center, dtype=np.float32)
        size = np.asarray(spec.size, dtype=np.float32)
        spread = np.maximum(size / 5.0, 0.05)
        xyz = center + self.rng.normal(scale=spread, size=(count, 3)).astype(np.float32)
        intensity = np.full((count, 1), spec.intensity, dtype=np.float32)
        return np.hstack([xyz, intensity])

    def _background_points(self, count: int) -> np.ndarray:
        x_min, x_max, y_min, y_max, z_min, z_max = self.roi_bounds
        xyz = np.stack(
            [
                self.rng.uniform(x_min, x_max, size=count),
                self.rng.uniform(y_min, y_max, size=count),
                self.rng.uniform(z_min, z_max, size=count),
            ],
            axis=1,
        ).astype(np.float32)
        intensity = self.rng.uniform(0.0, 0.35, size=(count, 1)).astype(np.float32)
        return np.hstack([xyz, intensity])

    def generate_scene(self, scene_index: int, num_objects: int = 3) -> SyntheticScene:
        objects = [self._sample_object() for _ in range(num_objects)]
        points = [self._background_points(self.num_background_points)]
        point_labels = [np.zeros(self.num_background_points, dtype=np.int32)]

        boxes = []
        class_ids = []

        for spec in objects:
            obj_points = self._object_points(spec)
            points.append(obj_points)
            point_labels.append(np.full(obj_points.shape[0], self.class_palette[spec.class_name], dtype=np.int32))
            boxes.append(
                np.array(
                    [
                        *spec.center,
                        *spec.size,
                        spec.yaw_deg,
                    ],
                    dtype=np.float32,
                )
            )
            class_ids.append(self.class_palette[spec.class_name])

        scene_points = np.concatenate(points, axis=0)
        scene_labels = np.concatenate(point_labels, axis=0)
        permutation = self.rng.permutation(scene_points.shape[0])
        scene_points = scene_points[permutation]
        scene_labels = scene_labels[permutation]

        metadata = {
            "num_objects": num_objects,
            "objects": [asdict(obj) for obj in objects],
            "roi_bounds": self.roi_bounds,
        }
        return SyntheticScene(
            scene_id=f"scene_{scene_index:05d}",
            points=scene_points.astype(np.float32),
            point_labels=scene_labels.astype(np.int32),
            boxes=np.stack(boxes, axis=0).astype(np.float32),
            class_ids=np.asarray(class_ids, dtype=np.int32),
            metadata=metadata,
        )

    def generate_dataset(self, num_scenes: int = 64, min_objects: int = 1, max_objects: int = 4) -> list[SyntheticScene]:
        scenes = []
        for index in range(num_scenes):
            num_objects = int(self.rng.integers(min_objects, max_objects + 1))
            scenes.append(self.generate_scene(index, num_objects=num_objects))
        return scenes

