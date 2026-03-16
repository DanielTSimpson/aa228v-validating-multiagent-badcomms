"""
Importance Sampling simulation script for Dec-POMDP multi-agent fire search
Run this file to start the simulation
"""
import numpy as np
from failtracker import FailTracker
from direct_estimation import run_simulation
from concurrent.futures import ProcessPoolExecutor
import config as cfg
import csv


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
        
        mu = cfg.MU_Q

        sigma = np.abs(mu * 0.5)
        sigma[sigma == 0] = 0.1 

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
                x_candidate[3] = np.clip(x_candidate[3], 0, 1.0)  # mu_wind
                x_candidate[4] = max(0.01, x_candidate[4]) # var_wind
                x_candidate[5] = np.clip(x_candidate[5], 0.0, 1.0)  # W_angle
                x_candidate[6] = max(0.01, x_candidate[6])# var_wind_angle_change

                population.append(x_candidate)
                futures.append(executor.submit(run_simulation, x = mu, trial_num=trial_counter, render=0, save_gif=False))

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