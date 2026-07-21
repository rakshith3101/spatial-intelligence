import numpy as np

class SLAMMapMemory:
    def __init__(self):
        # Memory banks for our SLAM system
        self.keyframes = []       # Stores past camera positions (X, Y, Z)
        self.map_points_3d = {}   # Dictionary mapping Feature IDs to their true 3D coordinates

    def add_keyframe(self, camera_pose):
        """Saves the flight computer's estimated position at a specific time step"""
        self.keyframes.append(camera_pose)
        print(f"[Frontend] Keyframe registered at position: X={camera_pose[0]:.2f}, Y={camera_pose[1]:.2f}, Z={camera_pose[2]:.2f}")

    def update_map_point(self, feature_id, observed_3d_point):
        """
        Simulates Backend Map Optimization.
        If a point is seen multiple times, we average the measurement to eliminate sensor noise.
        """
        if feature_id not in self.map_points_3d:
            # First time seeing this mountain peak / building corner
            self.map_points_3d[feature_id] = [observed_3d_point]
        else:
            # Point seen again! Append new observation for bundle adjustment optimization
            self.map_points_3d[feature_id].append(observed_3d_point)

    def optimize_map_backend(self):
        """Computes the optimized, cleaned-up coordinate matrix of the world"""
        optimized_points = {}
        for feat_id, observations in self.map_points_3d.items():
            # Basic optimization: Calculate the centroid of observations (Mean)
            optimized_points[feat_id] = np.mean(observations, axis=0)
        return optimized_points

# --- SIMULATING A FLIGHT RUN ---
slam_system = SLAMMapMemory()

# Time Step 1: Aircraft takes off, registers first keyframe
slam_system.add_keyframe(np.array([0.0, 0.0, 2.0]))
# It spots a radar tower on the ground (Feature #101)
slam_system.update_map_point(feature_id=101, observed_3d_point=np.array([5.1, 10.2, 0.1]))

# Time Step 2: Aircraft flies forward 3 meters
slam_system.add_keyframe(np.array([3.0, 0.0, 2.1]))
# It spots the same radar tower, but sensor noise gives a slightly different coordinate
slam_system.update_map_point(feature_id=101, observed_3d_point=np.array([4.9, 9.8, -0.0]))

# Run backend optimization to reconcile the map
clean_map = slam_system.optimize_map_backend()

print("\n==================================================")
print("             SLAM BACKEND MAP METRICS             ")
print("==================================================")
print(f"Total Unique Assets Mapped in Database: {len(clean_map)}")
print(f"Optimized Location for Radar Tower (ID 101):")
print(f"  X: {clean_map[101][0]:.2f}m | Y: {clean_map[101][1]:.2f}m | Z: {clean_map[101][2]:.2f}m")
print("==================================================")