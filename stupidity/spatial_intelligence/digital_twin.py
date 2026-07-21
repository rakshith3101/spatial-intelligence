import numpy as np
from typing import List, Dict, Tuple

class DigitalTwinEnvironment:
    """
    Maintains the 3D high-fidelity spatial asset matrix (The True World).
    """
    def __init__(self):
        # Coordinates of structural assets in the world: [X, Y, Z, Radius]
        self.structures: np.ndarray = np.array([
            [5.0,  5.0,  0.0, 2.0],  # Asset A: Control Tower Base
            [-3.0, 8.0,  0.0, 1.5],  # Asset B: Communications Mast
            [2.0,  12.0, 0.0, 3.0]   # Asset C: Maintenance Hangar
        ], dtype=np.float64)

class DroneSensorPayload:
    """
    Simulates a high-frequency spatial sensor payload (Depth Camera / LiDAR proxy).
    """
    def __init__(self, max_range: float = 15.0):
        self.max_range = max_range

    def capture_scan(self, drone_pose: np.ndarray, environment: DigitalTwinEnvironment) -> List[Dict]:
        """
        Executes a geometric range calculation between the drone's active position 
        and the structural assets within the Digital Twin environment.
        """
        detected_assets = []
        drone_pos = drone_pose[:3]
        
        for idx, asset in enumerate(environment.structures):
            asset_pos = asset[:3]
            radius = asset[3]
            
            # Calculate true Euclidean distance vector
            distance = np.linalg.norm(asset_pos - drone_pos) - radius
            
            # If the asset falls within the active sensor envelope, record the observation
            if distance <= self.max_range:
                # Calculate relative bearing angle to asset center
                relative_vector = asset_pos - drone_pos
                bearing = np.arctan2(relative_vector[1], relative_vector[0]) - drone_pose[3]
                
                detected_assets.append({
                    "asset_id": 100 + idx,
                    "measured_range": max(0.0, distance),
                    "relative_bearing_rad": float(np.mod(bearing + np.pi, 2 * np.pi) - np.pi)
                })
        return detected_assets

class SpatialAnalyticsEngine:
    """
    The core diagnostic processor. Evaluates state estimation errors 
    and builds the telemetry analysis log.
    """
    def __init__(self):
        self.telemetry_history: List[Dict] = []

    def log_telemetry(self, step: int, true_pose: np.ndarray, estimated_pose: np.ndarray, observations: List[Dict]):
        """Computes metric discrepancies between Ground Truth and system estimation."""
        localization_error = float(np.linalg.norm(true_pose[:3] - estimated_pose[:3]))
        
        self.telemetry_history.append({
            "step": step,
            "error_magnitude_meters": localization_error,
            "active_feature_count": len(observations)
        })

# --- EXECUTION INFRASTRUCTURE ---
if __name__ == "__main__":
    print("==================================================")
    print("     ENGAGING DRONE DIGITAL TWIN CORE ENGINE     ")
    print("==================================================")
    
    # Instantiate structural components
    world_twin = DigitalTwinEnvironment()
    sensor_ray = DroneSensorPayload(max_range=12.0)
    analytics = SpatialAnalyticsEngine()
    
    # Simulate a planned flight path matrix: [X, Y, Z, Yaw_Rad]
    flight_profile = [
        np.array([0.0, 0.0, 2.0, 0.0]),
        np.array([1.0, 2.0, 2.2, 0.1]),
        np.array([2.5, 5.0, 2.5, 0.2]),
        np.array([3.0, 8.0, 2.3, 0.3])
    ]
    
    for t_step, true_state in enumerate(flight_profile):
        # Simulate systematic hardware drift (Estimates degrade by 5cm per step)
        drift_vector = np.array([t_step * 0.05, t_step * 0.05, 0.0, 0.0])
        estimated_state = true_state + drift_vector
        
        # Capture sensor data based on the true physical location
        scans = sensor_ray.capture_scan(true_state, world_twin)
        
        # Run diagnostics
        analytics.log_telemetry(t_step, true_state, estimated_state, scans)
        
        print(f"Frame Execute [Step {t_step}]:")
        print(f"  True Coordinates : {true_state[:3]}")
        print(f"  Sensor Diagnostics: Detected {len(scans)} industrial structural reference points.")
        for scan in scans:
            print(f"    -> Asset ID [{scan['asset_id']}]: Range={scan['measured_range']:.2f}m | Bearing={scan['relative_bearing_rad']:.2f} rad")
        print("-" * 50)

    print("\n==================================================")
    print("         DIGITAL TWIN ANALYTICS SUMMARY           ")
    print("==================================================")
    for entry in analytics.telemetry_history:
        print(f"Step {entry['step']} | Divergence Error: {entry['error_magnitude_meters']:.4f} meters | Tracked Targets: {entry['active_feature_count']}")
    print("==================================================")