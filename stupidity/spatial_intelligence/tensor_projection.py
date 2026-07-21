import numpy as np
from typing import Tuple

class TensorProjectionNode:
    """
    High-performance spatial coordinate transformation engine.
    Handles batch transformations of 3D point tensors across coordinate frames
    for autonomous flight computers and digital twin alignments.
    """
    def __init__(self):
        pass

    def construct_se3_matrix(self, roll_deg: float, pitch_deg: float, yaw_deg: float, translation: np.ndarray) -> np.ndarray:
        """
        Constructs a formal 4x4 Homogeneous Transformation Matrix SE(3).
        Angles follow the aerospace convention (Tait-Bryan intrinsic rotations).
        """
        r, p, y = np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg)
        
        # Calculate individual axis rotation matrices
        R_x = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        R_y = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        R_z = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        
        # Combined rotation matrix: R = Rz * Ry * Rx
        R_combined = np.dot(R_z, np.dot(R_y, R_x))
        
        # Assemble the 4x4 SE(3) matrix
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_combined
        T[:3, 3] = translation
        
        return T

    def transform_point_cloud_tensor(self, point_tensor: np.ndarray, T_matrix: np.ndarray) -> np.ndarray:
        """
        Transforms an input feature tensor from a local coordinate frame to a target frame.
        Input point_tensor: Shape (N, 3) representing [X, Y, Z]
        Output point_tensor: Shape (N, 3) transformed into the target space
        """
        num_points = point_tensor.shape[0]
        
        # 1. Convert to homogeneous coordinates by appending a column of ones -> Shape (N, 4)
        ones = np.ones((num_points, 1), dtype=np.float64)
        homogeneous_points = np.hstack((point_tensor, ones))
        
        # 2. Execute vectorized batch transformation: (T_matrix * P^T)^T
        # Mathematically equivalent to multiplying every individual vector by the matrix
        transformed_homo = np.dot(homogeneous_points, T_matrix.T)
        
        # 3. Strip the homogeneous coordinate to return to standard 3D space -> Shape (N, 3)
        return transformed_homo[:, :3]

# --- PIPELINE RUNNER ---
if __name__ == "__main__":
    print("==================================================")
    print("      INITIALIZING SE(3) TENSOR PROJECTION        ")
    print("==================================================")
    
    projector = TensorProjectionNode()
    
    # Simulate a raw spatial feature tensor captured by an onboard sensor
    # Shape: (4 points, 3 coordinates: X, Y, Z relative to the sensor lens)
    raw_sensor_batch = np.array([
        [0.0, 5.0, 0.0],
        [1.0, 5.0, 0.2],
        [-1.0, 5.0, -0.2],
        [0.0, 6.0, 0.5]
    ], dtype=np.float64)
    
    # Flight State: Aircraft has pitched up 10 degrees, yawed 30 degrees, 
    # and moved to coordinates X=12.0m, Y=45.0m, Z=3.5m in the global Digital Twin map.
    T_aircraft_to_world = projector.construct_se3_matrix(
        roll_deg=0.0, 
        pitch_deg=10.0, 
        yaw_deg=30.0, 
        translation=np.array([12.0, 45.0, 3.5])
    )
    
    # Process the batch tensor transformation
    global_world_batch = projector.transform_point_cloud_tensor(raw_sensor_batch, T_aircraft_to_world)
    
    print("Transformation Matrix SE(3):\n", T_aircraft_to_world)
    print("\n--------------------------------------------------")
    print("Raw Input Coordinates (Local Sensor Frame):\n", raw_sensor_batch)
    print("\nProjected Output Coordinates (Global Digital Twin Frame):\n", global_world_batch)
    print("==================================================")