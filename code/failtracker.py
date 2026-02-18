import os
import csv
import uuid
import matplotlib.pyplot as plt
from datetime import datetime

class FailTracker:
    """
    Tracks simulation failures and metrics (entropy) over time.
    """
    def __init__(self):
        self.stuck_threshold = 10
        self.position_history = {} # {drone_id: [pos1, pos2, ...]}
        self.entropy_history = [] # List of {'time': t, 'drone_id_entropy': e, ...}
        self.failures = [] # List of failure messages
        self.stuck_drones = set() # Track which drones have already been flagged as stuck
        
    def update(self, drones, time_step):
        """
        Update tracker with current state of drones.
        Should be called every simulation step.
        """
        # Initialize data row for this time step
        step_data = {'time': time_step}
        
        for drone in drones:
            d_id = drone.drone_id
            
            # --- 1. Entropy Tracking ---
            entropy = drone.belief_state.get_entropy()
            step_data[f'drone_{d_id}_entropy'] = entropy
            
            # --- 2. Stuck Detection ---
            if d_id not in self.position_history:
                self.position_history[d_id] = []
            
            # Store tuple for easy comparison (numpy arrays aren't directly hashable)
            pos = tuple(drone.position)
            self.position_history[d_id].append(pos)
            
            # Keep only recent history defined by threshold
            if len(self.position_history[d_id]) > self.stuck_threshold:
                self.position_history[d_id].pop(0)
                
            # Check if stuck (history full and all positions identical)
            if len(self.position_history[d_id]) == self.stuck_threshold:
                first_pos = self.position_history[d_id][0]
                if all(p == first_pos for p in self.position_history[d_id]):
                    # Only report once per drone per stuck event (or just once per sim)
                    if d_id not in self.stuck_drones:
                        msg = f"STUCK: Drone {d_id} stuck at {first_pos} for {self.stuck_threshold} steps"
                        self.failures.append(msg)
                        self.stuck_drones.add(d_id)
                        print(f"!!! {msg} !!!")

        self.entropy_history.append(step_data)

    def check_timeout(self, fire_extinguished):
        """Call at the end of simulation to check if fire was found."""
        if not fire_extinguished:
            msg = "TIMEOUT: Fire not found/extinguished in time"
            self.failures.append(msg)
            print(f"!!! {msg} !!!")

    def save_data(self):
        """Saves entropy data to CSV in ../data folder."""
        # Determine paths relative to this file location
        # This file is in .../code/failtracker.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir) 
        data_dir = os.path.join(project_dir, 'data')
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir)
            except OSError as e:
                print(f"Error creating data directory: {e}")
                return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # --- Save Entropy CSV ---
        if self.entropy_history:
            filename = f"entropy_{timestamp}_{unique_id}.csv"
            filepath = os.path.join(data_dir, filename)
            
            # Get headers from the first entry
            keys = self.entropy_history[0].keys()
            
            try:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(self.entropy_history)
                print(f"Entropy data saved to: {filepath}")
            except IOError as e:
                print(f"Error saving entropy data: {e}")

        # --- Save Failures Report (if any) ---
        if self.failures:
            fail_filename = f"failures_{timestamp}_{unique_id}.txt"
            fail_path = os.path.join(data_dir, fail_filename)
            try:
                with open(fail_path, 'w') as f:
                    for fail in self.failures:
                        f.write(fail + "\n")
                print(f"Failure report saved to: {fail_path}")
            except IOError as e:
                print(f"Error saving failure report: {e}")

    def plot_entropy(self):
        """Generates a plot of entropy over time."""
        if not self.entropy_history:
            print("No entropy data to plot.")
            return

        times = [d['time'] for d in self.entropy_history]
        
        # Identify drone keys dynamically
        drone_keys = [k for k in self.entropy_history[0].keys() if 'entropy' in k]
        
        plt.figure(figsize=(10, 6))
        for key in drone_keys:
            values = [d[key] for d in self.entropy_history]
            label = key.replace('_entropy', '').replace('_', ' ').title()
            plt.plot(times, values, label=label)
            
        plt.xlabel('Time Step')
        plt.ylabel('Entropy (Uncertainty)')
        plt.title('Belief State Entropy Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()