import numpy as np
import random

class AutonomousDroneEnv:
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Resets the drone back to initial takeoff conditions"""
        self.altitude = 10.0   # Start high in the air (meters)
        self.velocity = 0.0    # Stationary at start (m/s)
        self.target_alt = 2.0  # Desired hover altitude (meters)
        self.time_steps = 0
        return np.array([self.altitude - self.target_alt, self.velocity])

    def step(self, action):
        """
        Executes a flight control command.
        Action 0: Decrease Thrust (Descend)
        Action 1: Maintain Thrust (Hover)
        Action 2: Increase Thrust (Ascend)
        """
        self.time_steps += 1
        
        # Apply physics based on the chosen action
        if action == 0:   acceleration = -2.0  # Falling faster
        elif action == 1: acceleration = -0.5  # Slight gravity sink
        else:             acceleration = 1.5   # Gaining lift
        
        # Euler integration for physics equations
        self.velocity += acceleration * 0.1
        self.altitude += self.velocity * 0.1
        
        # Calculate the state: [Distance error to target, current speed]
        error = self.altitude - self.target_alt
        next_state = np.array([error, self.velocity])
        
        # --- THE REWARD FUNCTION ---
        # Penalize being far away from the target zone
        reward = -abs(error)
        
        # Check termination conditions
        done = False
        if self.altitude <= 0.0:  # Drone crashed into the ground
            reward -= 50
            done = True
        elif abs(error) < 0.2:   # Successfully hovering at target!
            reward += 10
            if self.time_steps > 50: done = True
            
        if self.time_steps >= 100: # Timeout
            done = True
            
        return next_state, reward, done

# --- SIMULATING AN UNTRAINED RANDOM AGENT ---
env = AutonomousDroneEnv()
state = env.reset()
total_reward = 0
done = False

print("==================================================")
print("       AUTONOMOUS FLIGHT AGENT LOG (EPISODE 1)    ")
print("==================================================")

while not done:
    # A raw, untrained network just picks random actions
    random_action = random.choice([0, 1, 2]) 
    state, reward, done = env.step(random_action)
    total_reward += reward
    
    print(f"Step {env.time_steps:02d} | Altitude: {env.altitude:5.2f}m | Velocity: {env.velocity:5.2f}m/s | Step Reward: {reward:6.2f}")

print("==================================================")
print(f"Flight Concluded. Total Cumulative Reward: {total_reward:.2f}")
print("==================================================")