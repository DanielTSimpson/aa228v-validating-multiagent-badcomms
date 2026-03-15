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
from concurrent.futures import ProcessPoolExecutor
import config as cfg
import csv

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


def run_simulation(tracker, x:list = [], trial_num = 0, render=0, save_gif=False, save_data=True):
    """
    Run the complete Dec-POMDP multi-agent simulation
    Args:
        tracker: FailTracker object for getting and recording failure modes
        x: List of disturbances  
        render: Int to show the gridworld plot render (0 - Show nothing, 1 - Show status updates, 2 - Show all)
        save_gif: Boolean to save the set of plots as a .gif
        save_data: Boolean to save the tracking info to .csv
        
    Returns:
        None
    """
    print(f"Running Trial {trial_num}")
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
    droneA_old_pos = drones[0].position
    theta_A = np.random.uniform(0, np.pi / 2) # Choose a random initial angle from Uniform(0, 90deg)
    distance_A = np.random.normal(int(cfg.GRID_SIZE * x[1]), int(cfg.GRID_SIZE * x[2]**2)) # Choose a random initial distance from Normal(x[0], x[1])
    
    # Map polar coordinates to discrete grid relative to fire
    fire_x, fire_y = env.fire_pos
    new_x = int(x[0]*droneA_old_pos[0] + (1-x[0])*np.round(fire_x + distance_A * np.cos(theta_A))) # Bias the drone to the random position or radial position
    new_y = int(x[0]*droneA_old_pos[1] + (1-x[0])*np.round(fire_y + distance_A * np.sin(theta_A)))
    new_x = max(0, min(env.grid_size - 1, new_x))
    new_y = max(0, min(env.grid_size - 1, new_y))
    drones[0].position = np.array([new_x, new_y])
    drones[0].visited_cells = {(new_x, new_y)}
    drones[0].belief_state = Belief(env.grid_size)
    drones[0].fire_found = drones[0].observe()
    drones[0].history = [drones[0].state]


    # === Inject disturbances, x into Drone B ===
    droneB_old_pos = drones[1].position
    theta_B = np.random.uniform(0, np.pi / 2) # Choose a random initial angle from Uniform(0, 90deg)
    distance_B = np.random.normal(int(cfg.GRID_SIZE * x[1]), int(cfg.GRID_SIZE * x[2]**2)) # Choose a random initial distance from Normal(x[0], x[1])
    
    # Map polar coordinates to discrete grid relative to fire
    fire_x, fire_y = env.fire_pos
    new_x = int(x[0]*droneB_old_pos[0] + (1-x[0])*np.round(fire_x + distance_B * np.cos(theta_B))) # Bias the drone to the random position or radial position
    new_y = int(x[0]*droneB_old_pos[1] + (1-x[0])*np.round(fire_y + distance_B * np.sin(theta_B)))
    new_x = max(0, min(env.grid_size - 1, new_x))
    new_y = max(0, min(env.grid_size - 1, new_y))
    drones[1].position = np.array([new_x, new_y])
    drones[1].visited_cells = {(new_x, new_y)}
    drones[1].belief_state = Belief(env.grid_size)
    drones[1].fire_found = drones[1].observe()
    drones[1].history = [drones[1].state]

    # === Inject disturbances, x into Environment's Wind ===
    env.wind_speed = np.random.normal(x[3], x[4]**2) # Choose a random initial wind speed from Normal(x[3], x[4])
    env.wind_direction = x[5]*2*np.pi*np.random.random() + (1-x[5])*np.random.normal((theta_A + theta_B)/2, x[6]**2)

    # Main simulation loop
    for i in range(N):
        # Bias the Dynamic Wind every 10 time steps to push the drones away from the fire
        if i > 0 and i % 10 == 0:
            theta_A = np.arctan2(drones[0].position[1] - env.fire_pos[1], drones[0].position[0] - env.fire_pos[0])
            theta_B = np.arctan2(drones[1].position[1] - env.fire_pos[1], drones[1].position[0] - env.fire_pos[0])
            env.wind_direction = x[5]*2*np.pi*np.random.random() + (1-x[5])*np.random.normal((theta_A + theta_B)/2, x[6]**2)
            if render == 1:
                print(f"\tWind changed direction to {env.wind_direction*180/np.pi:.1f} degrees")

        # Render current state
        if render == 2 or save_gif:
            env.render(drones)
            if render == 2:
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
            if render == 1:
                print(f"\tFire extinguished!")
                if render == 2:
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
            if render == 1:
                print("\tFAILURE: Drones got Stuck")
            break

    if failure_mode == 2 and render == 1:
        print("\tFAILURE: Exceeded max sim time")

    # Calculate Total Cost (MAX_BUDGET - remaining budget)
    # We use the minimum remaining budget to represent the highest cost incurred by any agent
    final_min_budget = min(d.budget for d in drones)
    total_cost = cfg.MAX_BUDGET - final_min_budget
    total_time = time_to_obj*dt
    stuck_count = max(d.stuck_count for d in drones)
    
    if save_gif:
        gif_fps = int(2.0 / cfg.RENDER_PAUSE) if cfg.RENDER_PAUSE > 0 else 10
        env.save_gif(f"simulation_trial_{trial_num}.gif", fps=gif_fps)

    if save_data:
        tracker.log_failure(trial_num, failure_mode, total_cost, time_to_obj, total_comms)
    
    env.close()
    return [total_cost, total_time, stuck_count]


if __name__ == '__main__':
    tracker = FailTracker()
    save_data = False

    trial_counter = 0
    failure_modes = ["Total Cost", "Total Time", "Stuck Count"]

    executor = ProcessPoolExecutor()
    for mode_idx, mode_name in enumerate(failure_modes):
        print(f"\n{'='*40}")
        print(f"Optimizing for {mode_name}")
        print(f"{'='*40}")
        
        # Initial, nominal parameters
        # Order: [W_dist, mu_dist, var_dist, mu_wind, var_wind, W_angle, var_wind_angle_change]
        mu_p = np.array([
            1.0,                            # W_dist
            0.50,                           # mu_dist
            0.50,                           # var_dist
            0.25,                           # mu_wind
            0.01,                           # var_wind
            1,                              # W_angle
            1.0**2                          # var_wind_angle_change
        ])

        # Initial, "expert opinion" parameters
        # Order: [W_dist, mu_dist, var_dist, mu_wind, var_wind, W_angle, var_wind_angle_change]
        mu_q = np.array([
            0,                              # W_dist
            0.90,                           # mu_dist
            0.10,                           # var_dist
            0.3,                            # mu_wind
            0.01,                           # var_wind
            0,                              # W_angle
            0.01                            # var_wind_angle_change
        ])
        
        mu = mu_p

        # Initial covariance (std dev for exploration, set to 50% of mean initially)
        sigma = np.abs(mu * 0.5)
        sigma[sigma == 0] = 0.1 

        # Initialize CSV file for this failure mode
        csv_filename = f"AIS_params_{mode_name.replace(' ', '_')}.csv"
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Iteration', 'W_dist', 'mu_dist', 'var_dist', 'mu_wind', 'var_wind', 'W_angle', 'var_wind_angle_change']) # Header
            writer.writerow([1] + mu.tolist()) # Initial parameters (Iteration 1)

        num_iterations = 100
        samples_per_iter = 20
        num_elites = 5

        for i in range(num_iterations):
            population = []
            scores = []
            futures = []

            for j in range(samples_per_iter):
                trial_counter += 1
                # Sample parameters and enforce constraints
                x_candidate = np.random.normal(mu, sigma)
                x_candidate[0] = np.clip(x_candidate[0], 0, 1)   # W_dist
                x_candidate[1] = np.clip(x_candidate[1], 0, 1)  # mu_dist
                x_candidate[2] = np.clip(x_candidate[2], 0.01, 1) # var_dist
                x_candidate[3] = np.clip(x_candidate[3], 0, 0.3)  # mu_wind
                x_candidate[4] = max(0.01, x_candidate[4]) # var_wind
                x_candidate[5] = np.clip(x_candidate[5], 0.0, 1.0)  # W_angle
                x_candidate[6] = max(0.01, x_candidate[6])# var_wind_angle_change

                population.append(x_candidate)
                # Submit simulation to process pool. Pass None for tracker since save_data=False
                futures.append(executor.submit(run_simulation, None, x_candidate.tolist(), trial_counter, 0, False, False))

            # Collect results
            results = [f.result() for f in futures]
            scores = [res[mode_idx] for res in results]

            # Select Elites (Highest Score = Best for finding failure)
            population = np.array(population)
            scores = np.array(scores)
            elite_indices = np.argsort(scores)[::-1][:num_elites]
            elites = population[elite_indices]

            # Update parameters (Smoothing with alpha)
            alpha = 0.7
            prev_mu = mu.copy()
            mu = alpha * np.mean(elites, axis=0) + (1 - alpha) * mu
            sigma = alpha * np.std(elites, axis=0) + (1 - alpha) * sigma
            
            # Save updated parameters to CSV
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i + 2] + mu.tolist())
            
            print(f"Iter {i+1}: Max Score={np.max(scores):.2f}, Mean Score={np.mean(scores):.2f}")

            # Check for convergence: exit if all parameters changed by less than 0.0001
            if np.all(np.abs(mu - prev_mu) < 0.0001):
                print(f"Convergence reached at iteration {i+1} (delta < 0.0001).")
                break
        
        print(f"Best Params for {mode_name}: {mu}")
    executor.shutdown()