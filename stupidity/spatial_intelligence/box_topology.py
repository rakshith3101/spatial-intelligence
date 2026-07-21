import numpy as np
from typing import Dict, Tuple

class BoundingBoxTopology3D:
    """
    Simulates the performance analysis layer of a 3D Deep Learning Detection Head.
    Validates predicted 7-parameter Oriented Bounding Boxes against Ground Truth.
    """
    def __init__(self):
        pass

    def calculate_axis_aligned_3d_iou(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        """
        Computes the 3D Intersection over Union (IoU) for axis-aligned representations.
        Input box format: [x, y, z, w, l, h]
        """
        # 1. Determine the spatial coordinate boundaries of both bounding boxes
        ax_min, ax_max = box_a[0] - box_a[3]/2, box_a[0] + box_a[3]/2
        ay_min, ay_max = box_a[1] - box_a[4]/2, box_a[1] + box_a[4]/2
        az_min, az_max = box_a[2] - box_a[5]/2, box_a[2] + box_a[5]/2

        bx_min, bx_max = box_b[0] - box_b[3]/2, box_b[0] + box_b[3]/2
        by_min, by_max = box_b[1] - box_b[4]/2, box_b[1] + box_b[4]/2
        bz_min, bz_max = box_b[2] - box_b[5]/2, box_b[2] + box_b[5]/2

        # 2. Compute the intersecting boundaries
        inter_x_min = max(ax_min, bx_min)
        inter_x_max = min(ax_max, bx_max)
        inter_y_min = max(ay_min, by_min)
        inter_y_max = min(ay_max, by_max)
        inter_z_min = max(az_min, bz_min)
        inter_z_max = min(az_max, bz_max)

        # 3. Calculate volumes
        inter_w = max(0.0, inter_x_max - inter_x_min)
        inter_l = max(0.0, inter_y_max - inter_y_min)
        inter_h = max(0.0, inter_z_max - inter_z_min)
        
        intersection_volume = inter_w * inter_l * inter_h
        
        volume_a = box_a[3] * box_a[4] * box_a[5]
        volume_b = box_b[3] * box_b[4] * box_b[5]
        
        union_volume = volume_a + volume_b - intersection_volume
        
        if union_volume == 0:
            return 0.0
            
        return float(intersection_volume / union_volume)

# --- RUN PERFORMANCE ANALYSIS DIAGNOSTICS ---
if __name__ == "__main__":
    print("==================================================")
    print("      INITIALIZING 3D BOX REGRESSION ANALYSIS     ")
    print("==================================================")

    evaluator = BoundingBoxTopology3D()

    # Ground Truth State Vector: An inspected drone target located at coordinates:
    # X=5.0m, Y=12.0m, Z=2.5m. Size: Width=1.2m, Length=1.2m, Height=0.5m
    ground_truth_box = np.array([5.0, 12.0, 2.5, 1.2, 1.2, 0.5])

    # Simulation Node A: A highly accurate model prediction box
    predicted_box_good = np.array([5.05, 12.02, 2.48, 1.2, 1.2, 0.5])

    # Simulation Node B: A drifting, inaccurate model prediction box
    predicted_box_poor = np.array([5.40, 12.50, 2.80, 1.2, 1.2, 0.5])

    # Compute Metrics
    iou_good = evaluator.calculate_axis_aligned_3d_iou(ground_truth_box, predicted_box_good)
    iou_poor = evaluator.calculate_axis_aligned_3d_iou(ground_truth_box, predicted_box_poor)

    print("Target Ground Truth State Vector :", ground_truth_box)
    print("--------------------------------------------------")
    print(f"Prediction Variant A (Accurate Model Candidate):")
    print(f"  -> Predicted Tensor values: {predicted_box_good}")
    print(f"  -> Calculated Overlap Metric: 3D IoU = {iou_good:.4f}")
    
    print(f"\nPrediction Variant B (Drifting Model Candidate):")
    print(f"  -> Predicted Tensor values: {predicted_box_poor}")
    print(f"  -> Calculated Overlap Metric: 3D IoU = {iou_poor:.4f}")
    print("==================================================")