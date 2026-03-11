"""
Importance Sampling simulation script for Dec-POMDP multi-agent fire search
Run this file to start the simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from drone import Drone
from environment import SearchEnv
from failtracker import FailTracker
from belief import Belief
import config as cfg

def initialize_drones(num_drones, env, window_size):
    """Initialize drones at random positions that don't see the fire initially
    
    Args:
        num_drones: number of drones to create
        env: SearchEnv object
        window_size: the drone's observation window size
        
    Returns:
        list: list of Drone objects
    """
    drones = []

    for drone_id in range(num_drones):
        drone = Drone(env)

        # If drone observes the fire upon creation, reshuffle
        while drone.fire_found:
            drone = Drone(env)

        drone.drone_id = drone_id
        drone.window_size = window_size
        drone.gamma = cfg.GAMMA
        drone.exploration_bonus = cfg.EXPLORATION_BONUS 
        drone.movement_cost = cfg.MOVEMENT_COST
        drone.comm_cost = cfg.COMMUNICATION_COST
        drone.time_cost = cfg.TIME_COST    

        drones.append(drone)

    return drones


def run_simulation(tracker, x:list = [], trial_num = 0, render=False, save_gif=False):
    """
    Run the complete Dec-POMDP multi-agent simulation
    Args:
        tracker: FailTracker object for getting and recording failure modes
        x: List of disturbances  
        render: Boolean to show the gridworld plot render
        save_gif: Boolean to save the set of plots as a .gif
        
    Returns:
        None
    """
    # Initialize simulation parameters
    t_0 = cfg.INITIAL_TIME
    dt = cfg.TIME_STEP
    t_f = cfg.MAX_SIMULATION_TIME
    render_pause = cfg.RENDER_PAUSE if render else 0.0
    N = int((t_f - t_0) / dt)

    # Initialize environment
    env = SearchEnv()
    env.fire_pos = (1, 1) # Place the fire at one corner as a fixed extreme point
    env.grid_size = cfg.GRID_SIZE
    env.wind_speed = cfg.WIND_SPEED
    env.wind_direction = cfg.WIND_DIRECTION
    if save_gif:
        env.record_frames = True

    # Initialize the drone(s)
    drone_window_size = cfg.OBSERVATION_WINDOW_SIZE
    drones = initialize_drones(cfg.NUM_DRONES, env, drone_window_size)

    failure_mode = 2 # Default to "Out of Time"
    time_to_obj = N
    total_comms = 0


    # === Inject disturbances, x into Drone A ===
    theta_A = np.random.uniform(0, np.pi / 2) # Choose a random initial angle from Uniform(0, 90deg)
    distance_A = np.random.normal(x[0], x[1]) # Choose a random initial distance from Normal(x[0], x[1])
    
    # Map polar coordinates to discrete grid relative to fire
    fire_x, fire_y = env.fire_pos
    new_x = int(np.round(fire_x + distance_A * np.cos(theta_A)))
    new_y = int(np.round(fire_y + distance_A * np.sin(theta_A)))
    new_x = max(0, min(env.grid_size - 1, new_x))
    new_y = max(0, min(env.grid_size - 1, new_y))
    drones[0].position = np.array([new_x, new_y])
    drones[0].visited_cells = {(new_x, new_y)}
    drones[0].belief_state = Belief(env.grid_size)
    drones[0].fire_found = drones[0].observe()
    drones[0].history = [drones[0].state]


    # === Inject disturbances, x into Drone B ===
    theta_B = np.random.uniform(0, np.pi / 2) # Choose a random initial angle from Uniform(0, 90deg)
    distance_B = np.random.normal(x[0], x[1]) # Choose a random initial distance from Normal(x[0], x[1])
    
    # Map polar coordinates to discrete grid relative to fire
    fire_x, fire_y = env.fire_pos
    new_x = int(np.round(fire_x + distance_B * np.cos(theta_B)))
    new_y = int(np.round(fire_y + distance_B * np.sin(theta_B)))
    new_x = max(0, min(env.grid_size - 1, new_x))
    new_y = max(0, min(env.grid_size - 1, new_y))
    drones[1].position = np.array([new_x, new_y])
    drones[1].visited_cells = {(new_x, new_y)}
    drones[1].belief_state = Belief(env.grid_size)
    drones[1].fire_found = drones[1].observe()
    drones[1].history = [drones[1].state]

    # === Inject disturbances, x into Environment's Wind ===
    env.wind_speed = np.random.normal(x[2], x[3]) # Choose a random initial wind speed from Normal(x[0], x[1])
    env.wind_direction = (theta_A + theta_B)/2

    # Main simulation loop
    for i in range(N):
        # Bias the Dynamic Wind every 10 time steps to push the drones away from the fire
        if i > 0 and i % 10 == 0:
            theta_A = np.arctan2(drones[0].position[1] - env.fire_pos[1], drones[0].position[0] - env.fire_pos[0])
            theta_B = np.arctan2(drones[1].position[1] - env.fire_pos[1], drones[1].position[0] - env.fire_pos[0])
            env.wind_direction = (theta_A + theta_B) * 0.5
            if render:
                print(f"*** Wind changed direction to {env.wind_direction*180/np.pi:.1f} degrees ***")

        # Render current state
        if render or save_gif:
            env.render(drones)
            if render:
                plt.pause(render_pause)
        
        # Check for budget failure (Mode 1)
        # If any drone runs out of budget, we consider it a failure for the team/mission
        min_budget = min(d.budget for d in drones)
        if min_budget <= 0:
            failure_mode = 1
            time_to_obj = i
            break
        
        # Check if fire is extinguished
        if env.fire_extinguished:
            failure_mode = 0
            time_to_obj = i
            if render:
                print(60*"=")
                print(60*"=")
                print(f"Fire extinguished! Showing final state...")
                print(60*"=")
                print(60*"=")
                for j in range(1):  # Show final state for 10 frames
                    env.render(drones)
                    plt.pause(5)
            break
        
        # Dec-POMDP decision making and execution
        packets = []
        for drone in drones:
            action = drone.decide_action_pomdp()
            packet = drone.action(action)
            if packet:
                packets.append(packet)
        
        total_comms += len(packets)
        
        # Exchange packets between drones
        for packet in packets:
            for drone in drones:
                if drone.drone_id != packet['sender_id']:
                    drone.receive_telemetry(packet)
        
        # Check for stuck failure (Mode 3)
        if any(d.stuck_count >= 10 for d in drones):
            failure_mode = 3
            time_to_obj = i
            if render:
                print(60*"=")
                print("FAILURE: Drones got Stuck")
                print(60*"=")
            break

    if failure_mode == 2 and render:
        print(60*"=")
        print(60*"=")
        print("FAILURE: Exceeded max sim time")
        print(60*"=")
        print(60*"=")

    # Calculate Total Cost (MAX_BUDGET - remaining budget)
    # We use the minimum remaining budget to represent the highest cost incurred by any agent
    final_min_budget = min(d.budget for d in drones)
    total_cost = cfg.MAX_BUDGET - final_min_budget
    
    if save_gif:
        gif_fps = int(2.0 / cfg.RENDER_PAUSE) if cfg.RENDER_PAUSE > 0 else 10
        env.save_gif(f"simulation_trial_{trial_num}.gif", fps=gif_fps)

    tracker.log_failure(trial_num, failure_mode, total_cost, time_to_obj, total_comms)
    env.close()


if __name__ == '__main__':
    tracker = FailTracker()
    mu_dist = (int) (cfg.GRID_SIZE * 0.80)  # Average drone spawn distance away from the fire
    var_dist = 1                            # Variance of drone spawn distance away from the fire
    mu_wind = 0.25                          # Average wind magnitude
    var_wind = 0.0625                       # Variance of wind magnitude 
    x = [mu_dist, np.pow(var_dist, 2), mu_wind, np.pow(var_wind,2)] # Set of fuzzed variables
    run_simulation(tracker, x, render=True, save_gif=False)