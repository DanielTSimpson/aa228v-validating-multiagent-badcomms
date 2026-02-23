"""
Main simulation script for Dec-POMDP multi-agent fire search
Run this file to start the simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from drone import Drone
from environment import SearchEnv
from failtracker import FailTracker
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


def run_simulation(trial_num, tracker, render=False):
    """
    Run the complete Dec-POMDP multi-agent simulation
    """
    # Initialize simulation parameters
    t_0 = cfg.INITIAL_TIME
    dt = cfg.TIME_STEP
    t_f = cfg.MAX_SIMULATION_TIME
    render_pause = cfg.RENDER_PAUSE if render else 0.0

    N = int((t_f - t_0) / dt)
    #print(f"Max {N} time steps (Dec-POMDP with Value Iteration)")

    # Initialize environment
    env = SearchEnv()
    env.grid_size = cfg.GRID_SIZE
    env.wind_speed = cfg.WIND_SPEED
    env.wind_direction = cfg.WIND_DIRECTION

    # Initialize the drone(s)
    drone_window_size = cfg.OBSERVATION_WINDOW_SIZE
    drones = initialize_drones(cfg.NUM_DRONES, env, drone_window_size)

    failure_mode = 2 # Default to "Out of Time"
    time_to_obj = N
    total_comms = 0

    # Main simulation loop
    for i in range(N):
        # Dynamic wind: 25% chance to change every 10 timesteps
        if i > 0 and i % 10 == 0 and np.random.random() < 0.25:
            env.wind_direction = 2 * np.pi * np.random.random()
            if render:
                print(f"*** Wind changed direction to {env.wind_direction*180/np.pi:.1f} degrees ***")

        # Render current state
        if render:
            env.render(drones)
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
    
    tracker.log_failure(trial_num, failure_mode, total_cost, time_to_obj, total_comms)
    env.close()


if __name__ == '__main__':
    tracker = FailTracker()
    for i in range(1, 101):
        print(f"Running Trial {i}...")
        run_simulation(i, tracker, render=True)