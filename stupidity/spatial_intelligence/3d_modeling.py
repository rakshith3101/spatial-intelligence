import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_3d_box_corners(cx, cy, cz, w, h, d, yaw):
    """
    Computes the 8 corners of a 3D bounding box given its center, 
    dimensions, and yaw rotation angle.
    """
    # 1. Define local 8 corners of the box centered at (0,0,0)
    x_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]
    y_corners = [h/2, -h/2, -h/2, h/2, w/2, -h/2, -h/2, h/2] # Assuming height axis
    z_corners = [d/2, d/2, d/2, d/2, -d/2, -d/2, -d/2, -d/2]
    
    corners_local = np.vstack([x_corners, y_corners, z_corners])
    
    # 2. Create rotation matrix around Y-axis (Yaw in camera coordinate frame)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ])
    
    # 3. Rotate and translate corners to world location
    corners_rotated = np.dot(R, corners_local)
    corners_3d = corners_rotated + np.array([[cx], [cy], [cz]])
    
    return corners_3d.T

# --- SIMULATING NETWORK INFERENCE ---

# Imagine your Deep Learning Model detects an enemy fighter jet on the tarmac.
# Network outputs: Center=(1.5m right, -0.5m up, 12m away), Dim=(10m wide, 4m high, 15m deep), Yaw=45 degrees
network_prediction = {
    'center': (1.5, -0.5, 12.0),
    'dimensions': (10.0, 4.0, 15.0),
    'yaw': np.radians(45.0)
}

cx, cy, cz = network_prediction['center']
w, h, d = network_prediction['dimensions']
yaw = network_prediction['yaw']

box_corners = compute_3d_box_corners(cx, cy, cz, w, h, d, yaw)

# --- VISUALIZE SPATIAL PERCEPTION ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the 8 corners
ax.scatter(box_corners[:, 0], box_corners[:, 2], box_corners[:, 1], c='r', s=50, label='Predicted Object Vertices')

# Index mapping to draw the lines connecting the 8 corners of the cube
edges = [
    [0,1], [1,2], [2,3], [3,0], # Top face
    [4,5], [5,6], [6,7], [7,4], # Bottom face
    [0,4], [1,5], [2,6], [3,7]  # Verticals
]

for edge in edges:
    ax.plot(box_corners[edge, 0], box_corners[edge, 2], box_corners[edge, 1], 'b-')

ax.set_title("Autonomous Target Vectoring: 3D Bounding Box Projection", fontsize=12)
ax.set_xlabel("X (Lateral Matrix - Meters)")
ax.set_ylabel("Z (Depth Matrix - Meters)")
ax.set_zlabel("Y (Altitude Matrix - Meters)")
ax.legend()
plt.show()