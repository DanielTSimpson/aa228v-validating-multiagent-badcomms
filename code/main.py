from direct_estimation import run_simulation
import numpy as np
import config as cfg

if __name__ == '__main__':
    mu_q = cfg.MU_Q
    mu_p = cfg.MU_P
    
    run_simulation(x = mu_p, render=2, save_gif=True)