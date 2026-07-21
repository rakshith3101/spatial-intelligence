import numpy as np
from typing import Tuple, Dict

class VoxelGridEncoder:
    """
    Industrial preprocessing engine that discretizes irregular 3D spatial coordinate
    matrices into structured volumetric tensors for 3D Deep Learning models.
    """
    def __init__(self, voxel_size: Tuple[float, float, float] = (0.2, 0.2, 0.2),
                 spatial_bounds: Tuple[float, float, float, float, float, float] = (-5.0, 5.0, 0.0, 10.0, -2.0, 2.0)):
        """
        Args:
            voxel_size: The physical size of each voxel cuboid (dx, dy, dz) in meters.
            spatial_bounds: The active tracking volume (Min_X, Max_X, Min_Y, Max_Y, Min_Z, Max_Z).
                            Points outside these bounds are cropped to maintain constant input dimensions.
        """
        self.vx, self.vy, self.vz = voxel_size
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = spatial_bounds

        # Calculate the exact dimensions of the output voxel grid tensor
        self.grid_width = int(np.round((self.x_max - self.x_min) / self.vx))
        self.grid_depth = int(np.round((self.y_max - self.y_min) / self.vy))
        self.grid_height = int(np.round((self.z_max - self.z_min) / self.vz))

    def encode_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Transforms a raw point matrix [N, 4] -> [X, Y, Z, Intensity] into a 
        structured 3D volumetric density tensor of shape (Width, Depth, Height).
        """
        # 1. Filter out points that fall outside our designated spatial tracking bounds
        mask = (
            (point_cloud[:, 0] >= self.x_min) & (point_cloud[:, 0] < self.x_max) &
            (point_cloud[:, 1] >= self.y_min) & (point_cloud[:, 1] < self.y_max) &
            (point_cloud[:, 2] >= self.z_min) & (point_cloud[:, 2] < self.z_max)
        )
        filtered_points = point_cloud[mask]

        if filtered_points.shape[0] == 0:
            return np.zeros((self.grid_width, self.grid_depth, self.grid_height), dtype=np.float32)

        # 2. Compute voxel grid coordinate indices using fast vectorized division
        voxel_indices_x = np.floor((filtered_points[:, 0] - self.x_min) / self.vx).astype(np.int32)
        voxel_indices_y = np.floor((filtered_points[:, 1] - self.y_min) / self.vy).astype(np.int32)
        voxel_indices_z = np.floor((filtered_points[:, 2] - self.z_min) / self.vz).astype(np.int32)

        # 3. Populate our dense feature tensor
        # Here we calculate the point density (how many points fall into each voxel cell)
        dense_tensor = np.zeros((self.grid_width, self.grid_depth, self.grid_height), dtype=np.float32)
        
        for i in range(filtered_points.shape[0]):
            u, v, w = voxel_indices_x[i], voxel_indices_y[i], voxel_indices_z[i]
            # Ensure indices remain safely within boundaries
            if (0 <= u < self.grid_width) and (0 <= v < self.grid_depth) and (0 <= w < self.grid_height):
                dense_tensor[u, v, w] += 1.0 # Increment density count

        return dense_tensor

# --- PIPELINE DEMONSTRATION ---
if __name__ == "__main__":
    print("==================================================")
    print("      INITIALIZING HIGH-SPEED VOXEL ENCODER       ")
    print("==================================================")

    # Simulate an irregular point cloud of a drone flying through a narrow corridor
    # 5,000 points scattered dynamically
    np.random.seed(42)
    num_points = 5000
    mock_positions = np.random.uniform(-6, 6, size=(num_points, 3))
    mock_intensities = np.random.uniform(0.1, 1.0, size=(num_points, 1))
    raw_cloud = np.hstack((mock_positions, mock_intensities))

    # Initialize the voxelizer with 20cm resolution grids
    voxelizer = VoxelGridEncoder(
        voxel_size=(0.20, 0.20, 0.20),
        spatial_bounds=(-4.0, 4.0, 0.0, 8.0, -1.0, 1.0)
    )

    print(f"Target Tensor Resolution Dimensions: {voxelizer.grid_width}x{voxelizer.grid_depth}x{voxelizer.grid_height}")
    print(f"Processing raw points tensor of shape: {raw_cloud.shape}")
    print("--------------------------------------------------")

    # Encode raw coordinates into structured 3D spatial tensor
    d_feature_tensor = voxelizer.encode_point_cloud(raw_cloud)

    print("Encoding Complete.")
    print(f"Output Volumetric Tensor Shape: {d_feature_tensor.shape}")
    print(f"Total Non-Empty Spatial Voxels : {np.count_nonzero(d_feature_tensor)}")
    print(f"Maximum Point Density inside single Voxel Cell: {np.max(d_feature_tensor)}")
    print("==================================================")