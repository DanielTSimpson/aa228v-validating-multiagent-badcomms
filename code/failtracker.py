import csv
import os

class FailTracker:
    def __init__(self):
        # Save in a 'data' subdirectory relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(base_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.filepath = os.path.join(self.data_dir, 'simulation_results.csv')
        self.headers = ['Trial #', 'Failure Mode', 'Total Cost', 'Time to Objective', 'Num Comms']
        
        # Initialize file with headers (overwrite for new batch)
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_failure(self, trial_num, failure_mode, total_cost, time_to_objective, num_comms):
        """
        Logs a single trial result.
        """
        mode_map = {
            0: "No Failure",
            1: "Budget runs out",
            2: "Out of Time",
            3: "Drones got Stuck"
        }
        
        mode_str = mode_map.get(failure_mode, "Unknown")
        
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([trial_num, mode_str, f"{total_cost:.2f}", time_to_objective, num_comms])