import numpy as np
"""
Configuration file for Dec-POMDP multi-agent simulation
Centralized place for all simulation parameters
"""

# ===== Simulation (main) parameters ===== 
INITIAL_TIME = 0.0
TIME_STEP = 0.05
MAX_SIMULATION_TIME = 15.0
MAX_BUDGET = 1280
RENDER_PAUSE = 0.2

#  ===== Environment parameters ===== 
GRID_SIZE = 50  # Size of the NxN grid
WIND_SPEED = 0.25 # Probability of agents drifting after an action
WIND_DIRECTION = 2*np.pi*np.random.random() # Direction of the wind in radians (CCW)

# ===== Drone parameters ===== 
NUM_DRONES = 2
OBSERVATION_WINDOW_SIZE = 3

# ===== Dec-POMDP parameters ===== 
GAMMA = 0.95
EXPLORATION_BONUS = 10.0  # Bonus reward for exploring new cells, promotes active exploration of new cells

# === Cost parameters ===
COMMUNICATION_COST = 5.0
MOVEMENT_COST = 1.0
TIME_COST = 3.0