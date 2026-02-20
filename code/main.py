"""
Main simulation script for Dec-POMDP multi-agent fire search
Run this file to start the simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from drone import Drone
from environment import SearchEnv
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


def run_simulation():
    """
    Run the complete Dec-POMDP multi-agent simulation
    """
    # Initialize simulation parameters
    t_0 = cfg.INITIAL_TIME
    dt = cfg.TIME_STEP
    t_f = cfg.MAX_SIMULATION_TIME
    render_pause = cfg.RENDER_PAUSE

    N = int((t_f - t_0) / dt)
    print(f"Max {N} time steps (Dec-POMDP with Value Iteration)")

    # Initialize environment
    env = SearchEnv()
    env.grid_size = cfg.GRID_SIZE
    env.wind_speed = cfg.WIND_SPEED
    env.wind_direction = cfg.WIND_DIRECTION

    # Initialize the drone(s)
    drone_window_size = cfg.OBSERVATION_WINDOW_SIZE
    drones = initialize_drones(cfg.NUM_DRONES, env, drone_window_size)

    # Main simulation loop
    for i in range(N):
        # Dynamic wind: 25% chance to change every 10 timesteps
        if i > 0 and i % 10 == 0 and np.random.random() < 0.25:
            env.wind_direction = 2 * np.pi * np.random.random()
            print(f"*** Wind changed direction to {env.wind_direction*180/np.pi:.1f} degrees ***")

        # Render current state
        env.render(drones)
        plt.pause(render_pause)
        
        # Check if fire is extinguished
        if env.fire_extinguished:
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
        
        # Exchange packets between drones
        for packet in packets:
            for drone in drones:
                if drone.drone_id != packet['sender_id']:
                    drone.receive_telemetry(packet)

    else:
        print(60*"=")
        print(60*"=")
        print("FAILURE: Exceeded max sim time")
        print(60*"=")
        print(60*"=")

    env.close()


if __name__ == '__main__':
    for i in range(1, 10):
        run_simulation()