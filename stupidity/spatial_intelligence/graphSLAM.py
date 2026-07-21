import numpy as np
import matplotlib.pyplot as plt

class FactorGraphSLAM:
    def __init__(self):
        self.poses = []         # True historical states [X, Y] of the aircraft
        self.odometry_factors = [] # Relative changes detected by flight sensors (dx, dy)
        self.loop_closures = [] # Constraints connecting non-consecutive poses
        
    def add_aircraft_step(self, true_pose, measured_odom):
        """Registers a new node and its connecting odometry spring factor"""
        self.poses.append(np.array(true_pose))
        self.odometry_factors.append(np.array(measured_odom))
        
    def inject_loop_closure(self, from_idx, to_idx, constraint_vector):
        """Creates a spatial anchor constraint between two distant points in time"""
        self.loop_closures.append((from_idx, to_idx, np.array(constraint_vector)))
        print(f"[Loop Closure Detect] Connecting Pose {from_idx} directly back to Pose {to_idx}!")

    def optimize_trajectory(self):
        """
        Simulates a Non-Linear Least Squares Optimization Backend (Bundle Adjustment).
        Distributes accumulated mathematical drift evenly across the entire network graph.
        """
        n = len(self.poses)
        estimated_poses = [self.poses[0].copy()]
        
        # 1. Reconstruct raw drifting flight path from noisy odometry
        for i in range(1, n):
            estimated_poses.append(estimated_poses[-1] + self.odometry_factors[i-1])
            
        drifting_trajectory = np.array(estimated_poses)
        
        # 2. Apply Graph Adjustment optimization if a Loop Closure anchor is present
        optimized_trajectory = drifting_trajectory.copy()
        if self.loop_closures:
            from_idx, to_idx, constraint = self.loop_closures[0]
            # Calculate the overall drift error at the closure point
            actual_loop_error = optimized_trajectory[from_idx] - (optimized_trajectory[to_idx] + constraint)
            
            # Smoothly distribute the correction back through the historical nodes
            for j in range(to_idx + 1, from_idx + 1):
                blend_factor = (j - to_idx) / (from_idx - to_idx)
                optimized_trajectory[j] -= actual_loop_error * blend_factor
                
        return drifting_trajectory, optimized_trajectory

# --- INITIALIZE INDUSTRIAL FLIGHT TEST LOOP ---
slam_graph = FactorGraphSLAM()

# Simulate a drone flying a perfect square trajectory box: 
# Start at (0,0) -> (0,10) -> (10,10) -> (10,0) -> back to (0,0)
true_path = [[0,0], [0,10], [10,10], [10,0], [0,0]]

# Inject systematic IMU/Odometry drift sensor noise (+0.5m creep per step)
noisy_odom = [[0, 10.5], [10.5, 0], [0, -10.5], [-10.5, 0]]

# Load data nodes into graph memory
for i in range(len(true_path)):
    odom = noisy_odom[i] if i < len(noisy_odom) else [0,0]
    slam_graph.add_aircraft_step(true_path[i], odom)

# At the final step, the camera recognizes the initial takeoff landmark at (0,0)
slam_graph.inject_loop_closure(from_idx=4, to_idx=0, constraint_vector=[0, 0])

# Run the Graph Solver Optimization Backend
drifting_path, corrected_path = slam_graph.optimize_trajectory()

# --- PLOT THE REAL-TIME GRAPH OPTIMIZATION ---
plt.figure(figsize=(10, 6))
true_path = np.array(true_path)
plt.plot(true_path[:,0], true_path[:,1], 'g-', label='Ground Truth (Planned Path)', linewidth=2)
plt.plot(drifting_path[:,0], drifting_path[:,1], 'r--', label='Drifting Trajectory (No Loop Closure)', marker='x')
plt.plot(corrected_path[:,0], corrected_path[:,1], 'b-o', label='Optimized Graph Solution (SLAM Backend)')
plt.title("Industrial Spatial Intelligence: Pose-Graph Loop Closure Optimization")
plt.xlabel("X Grid Coordinates (Meters)")
plt.ylabel("Y Grid Coordinates (Meters)")
plt.legend()
plt.grid(True)
plt.show()