import numpy as np
from typing import Tuple, Dict, List

class VehicleState:
    """
    Represents the formal 2D kinematic state vector of the aircraft.
    State vector: [x, y, theta] where theta is the heading angle in radians.
    """
    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.position: np.ndarray = np.array([x, y], dtype=np.float64)
        self.theta: float = float(theta)

    def to_matrix(self) -> np.ndarray:
        """Converts the kinematic state to a homogeneous transformation matrix SE(2)."""
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        return np.array([
            [cos_t, -sin_t, self.position[0]],
            [sin_t,  cos_t, self.position[1]],
            [0.0,    0.0,    1.0]
        ], dtype=np.float64)

class AerospaceEnvironmentSimulator:
    """
    The deterministic physics engine. Maintains Ground Truth and generates 
    stochastic (noisy) sensor telemetry streams.
    """
    def __init__(self, process_noise_std: float = 0.05, sensor_noise_std: float = 0.2):
        self.ground_truth_state = VehicleState(x=0.0, y=0.0, theta=0.0)
        self.Q_std = process_noise_std  # Physical drift perturbation
        self.R_std = sensor_noise_std   # Electronic sensor noise

    def propagate_physics(self, forward_velocity: float, angular_velocity: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagates the true state via non-linear kinematic equations and 
        returns a noisy sensor observation vector.
        
        Kinematics:
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = omega
        """
        # 1. Update Ground Truth physics
        ds = forward_velocity * dt
        dtheta = angular_velocity * dt
        
        self.ground_truth_state.position[0] += ds * np.cos(self.ground_truth_state.theta)
        self.ground_truth_state.position[1] += ds * np.sin(self.ground_truth_state.theta)
        self.ground_truth_state.theta = (self.ground_truth_state.theta + dtheta) % (2 * np.pi)
        
        # 2. Generate Noisy Odometry Telemetry (Simulating IMU/Dead Reckoning)
        noisy_ds = ds + np.random.normal(0.0, self.Q_std)
        noisy_dtheta = dtheta + np.random.normal(0.0, self.Q_std * 0.1)
        
        control_telemetry = np.array([noisy_ds, noisy_dtheta])
        
        return self.ground_truth_state.position.copy(), control_telemetry

# --- PRODUCTION PIPELINE EXECUTOR ---
if __name__ == "__main__":
    print("==================================================")
    print("    INITIALIZING PROD-GRADE AEROSPACE SIMULATOR   ")
    print("==================================================")
    
    # Initialize simulation environment
    sim = AerospaceEnvironmentSimulator(process_noise_std=0.08, sensor_noise_std=0.15)
    dt_step = 0.1  # 10 Hz execution cycle
    
    print("System Status: Online. Executing flight matrix propagation loop...")
    print("--------------------------------------------------")
    
    # Simulate a steady forward flight sweep with a continuous banking turn
    for step in range(1, 6):
        true_pos, telemetry = sim.propagate_physics(forward_velocity=5.0, angular_velocity=0.2, dt=dt_step)
        
        print(f"Time Step {step} [dt={dt_step}s]:")
        print(f"  Ground Truth Location : X={true_pos[0]:.4f}m, Y={true_pos[1]:.4f}m")
        print(f"  Noisy IMU Telemetry   : Delta_S={telemetry[0]:.4f}m, Delta_Theta={telemetry[1]:.4f}rad")
    
    print("==================================================")