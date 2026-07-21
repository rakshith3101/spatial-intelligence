import numpy as np

class AerospaceSensorFusion:
    def __init__(self):
        # Initial state estimate: Velocity = 0.0 m/s
        self.velocity_estimate = 0.0
        # Estimation uncertainty (Error covariance)
        self.P = 1.0
        # Process Noise (How much we trust our physical physics model)
        self.Q = 0.05
        # Measurement Noise (How noisy our camera sensor is)
        self.R = 0.5
        
        # Scaling factor: For this altitude, assume 10 pixels roughly equals 1 meter
        self.PIXELS_PER_METER = 10.0

    def update_state(self, imu_acceleration, camera_pixel_shift, dt=0.1):
        """
        Fuses IMU and Camera data to estimate true velocity.
        dt = time step between frames (e.g., 0.1 seconds)
        """
        # 1. PREDICT STEP (Using Physics Model + IMU)
        # v = u + a*t
        self.velocity_estimate = self.velocity_estimate + (imu_acceleration * dt)
        self.P = self.P + self.Q  # Increase uncertainty because IMU drifts

        # 2. MEASUREMENT STEP (Convert camera pixel shifts to a mock velocity measurement)
        # Velocity = Distance / Time -> (pixels / scale) / time
        measured_distance = camera_pixel_shift / self.PIXELS_PER_METER
        measured_velocity = measured_distance / dt

        # 3. UPDATE STEP (The Kalman Magic)
        # Calculate the Kalman Gain (weighting factor between model and sensor)
        kalman_gain = self.P / (self.P + self.R)
        
        # Correct our velocity estimate using the camera measurement
        self.velocity_estimate = self.velocity_estimate + kalman_gain * (measured_velocity - self.velocity_estimate)
        
        # Update our uncertainty matrix
        self.P = (1 - kalman_gain) * self.P
        
        return self.velocity_estimate

# --- EXECUTION ---
fusion_engine = AerospaceSensorFusion()

# Let's pull your actual vertical pixel shift from your output: 28.67 pixels
camera_dy = 28.67 
# Imagine the IMU says we are accelerating forward at 2.5 m/s^2
imu_accel = 2.5   
# Time difference between frames (10 frames per second = 0.1s)
dt = 0.1          

true_velocity = fusion_engine.update_state(imu_accel, camera_dy, dt)

print("==================================================")
print("         EXTENDED KALMAN FILTER OUTPUT            ")
print("==================================================")
print(f"Raw Input - Camera Pixel Shift: {camera_dy:.2f} px")
print(f"Raw Input - IMU Acceleration:   {imu_accel:.2f} m/s²")
print("--------------------------------------------------")
print(f"FUSED ESTIMATED TRUE VELOCITY:  {true_velocity:.2f} meters/second")
print(f"In Knots (Aviation Standard):   {true_velocity * 1.94384:.2f} knots")
print("==================================================")