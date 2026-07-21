# import numpy as np
# import cv2

# # 1. Define the REAL WORLD 3D coordinates of our target (Landing Pad)
# # Let's say it's a 1-meter by 1-meter square pad flat on the ground (Z=0)
# # Coordinates are in meters: (X, Y, Z)
# landing_pad_3d = np.array([
#     [-0.5,  0.5, 0.0],  # Top-Left corner
#     [ 0.5,  0.5, 0.0],  # Top-Right corner
#     [ 0.5, -0.5, 0.0],  # Bottom-Right corner
#     [-0.5, -0.5, 0.0]   # Bottom-Left corner
# ], dtype=np.float32)

# # 2. Use a standard pre-calibrated Intrinasic Matrix (K) 
# # Assuming an HD camea with focal length ~800 pixels
# K = np.array([
#     [800.0,   0.0, 640.0],
#     [  0.0, 800.0, 360.0],
#     [  0.0,   0.0,   1.0]
# ], dtype=np.float32)

# # Zero lens distortion assumed for this clean tracking run
# dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# # 3. Simulate 2D pixel coordinates detected by our drone's computer vision system
# # As the drone approaches at an angle, the square looks warped on screen
# detected_pixels_2d = np.array([
#     [480, 200],  # Top-Left corner pixel
#     [800, 220],  # Top-Right corner pixel
#     [850, 500],  # Bottom-Right corner pixel
#     [430, 480]   # Bottom-Left corner pixel
# ], dtype=np.float32)

# # 4. SOLVE THE PNP GEOMETRY
# # cv2.solvePnP takes: 3D model, 2D pixels, K matrix, Distortion
# success, rvec, tvec = cv2.solvePnP(landing_pad_3d, detected_pixels_2d, K, dist_coeffs)

# if success:
#     # tvec is the translation vector [X, Y, Z] relative to the camera lens
#     X_dist, Y_dist, Z_dist = tvec.flatten()
    
#     # Convert the rotation vector (rvec) to a 3x3 Rotation Matrix (R)
#     R, _ = cv2.Rodrigues(rvec)
    
#     # Calculate absolute straight-line distance (Euclidean distance)
#     range_to_target = np.linalg.norm(tvec)
    
#     print("==================================================")
#     print("      SPATIAL NAVIGATION LOCALIZATION DATA        ")
#     print("==================================================")
#     print(f"Relative Position Coordinates (Meters):")
#     print(f"  X (Lateral Offset): {X_dist:.2f} meters")
#     print(f"  Y (Vertical Offset): {Y_dist:.2f} meters")
#     print(f"  Z (True Altitude/Range): {Z_dist:.2f} meters")
#     print("--------------------------------------------------")
#     print(f"TOTAL SLANT RANGE TO TARGET: {range_to_target:.2f} meters")
#     print("==================================================")
#     print("\nRotation Matrix (R) Orientation Alignment:")
#     print(R)
# else:
#     print("Spatial geometry tracking failed!")


import math
import numpy as np

def rotation_matrix_to_euler_angles(R):
    """
    Extracts Pitch, Roll, and Yaw (in degrees) from a 3x3 Rotation Matrix.
    Uses the standard aerospace robotics convention.
    """
    # Calculate Pitch (rotation around Y axis)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1] , R[2,2]) # Roll
        y = math.atan2(-R[2,0], sy)      # Pitch
        z = math.atan2(R[1,0] , R[0,0]) # Yaw
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # Convert radians to human-readable degrees
    return math.degrees(x), math.degrees(y), math.degrees(z)

# Your actual output rotation matrix from the PnP solver
R_output = np.array([
    [ 0.99615076, -0.0007808,   0.08765301],
    [ 0.05418789, -0.78051655, -0.62278214],
    [ 0.08890089,  0.62513464, -0.77746984]
])

roll, pitch, yaw = rotation_matrix_to_euler_angles(R_output)

print("==================================================")
print("          AIRCRAFT ATTITUDE TELEMETRY             ")
print("==================================================")
print(f"  ROLL  (Wing Bank Angle): {roll:.2f}°")
print(f"  PITCH (Nose Tilt Angle): {pitch:.2f}°")
print(f"  YAW   (Heading Angle):   {yaw:.2f}°")
print("==================================================")