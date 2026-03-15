from direct_estimation import run_simulation
import numpy as np

if __name__ == '__main__':
    mu_q = np.array([
        0,                              # W_dist
        0.90,                           # mu_dist
        0.10,                           # var_dist
        0.3,                            # mu_wind
        0.01,                           # var_wind
        0,                              # W_angle
        0.01                            # var_wind_angle_change
    ])
    mu_p = np.array([
        1.0,                            # W_dist
        0.50,                           # mu_dist
        0.50,                           # var_dist
        0.05,                           # mu_wind
        0.01,                           # var_wind
        1,                              # W_angle
        1.0**2                          # var_wind_angle_change
    ])
    
    run_simulation(x = mu_p, render=2, save_gif=False)