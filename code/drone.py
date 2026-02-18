"""
Drone module for Dec-POMDP multi-agent system
Implements intelligent decision-making for fire search and extinguish
"""
import numpy as np
from copy import deepcopy
from environment import SearchEnv
from belief import Belief     

class Drone():
    """
    Dec-POMDP Agent with belief state and value-based decision making
    """
    def __init__(self, environment: SearchEnv):
        self.drone_id = 0
        self.window_size = 0
        self.env = environment
        self.position = np.random.randint(0, self.env.grid_size, size=2)
        self.belief_state = Belief(self.env.grid_size)
        
        self.time = 0
        self.visited_cells = set() # Track the global visited cells
        self.visited_cells.add((self.x, self.y))
        self.steps_since_last_comm = 0
        self.last_action = None
        self.fire_found = False
        self.drifted = False
        self.history = []

        # POMDP Params
        self.gamma = 0.0
        self.exploration_bonus = 0.0 
        self.movement_cost = 0.0
        self.comm_cost = 0.0
        self.time_cost = 0.0
        
        self.fire_found = self.observe()
        if not self.history:
            self.history.append(self.state)
        
    @property
    def x(self):
        return self.position[0]
    
    @property
    def y(self):
        return self.position[1]

    @property
    def state(self):
        return [self.x, self.y, self.fire_found]


    def action(self, action):
        """Execute action and update state"""
        self.last_action = action
        x = self.x
        y = self.y
        telemetry_packet = None
        
        if action == 1:  # Up
            y = min(self.env.grid_size - 1, self.y + 1)
        elif action == 2:  # Down
            y = max(0, self.y - 1)
        elif action == 3:  # Left
            x = max(0, self.x - 1)
        elif action == 4:  # Right
            x = min(self.env.grid_size - 1, self.x + 1)
        elif action == 5: # Communicate
            telemetry_packet = self.create_telemetry_packet()
            self.steps_since_last_comm = 0
        elif action == 6: # Extinguish the Fire
            self.env.fire_extinguished = True

        self.drifted = False
        # Apply wind drift
        if np.random.random() < self.env.wind_speed:
            self.drifted = True
            wind_dx = int(np.round(np.cos(self.env.wind_direction)))
            wind_dy = int(np.round(np.sin(self.env.wind_direction)))
            print(f"Drone {self.drone_id} is drifting along {self.env.wind_direction*180/np.pi} degrees ({wind_dx},{wind_dy})\n")
            x = max(0, min(self.env.grid_size - 1, x + wind_dx))
            y = max(0, min(self.env.grid_size - 1, y + wind_dy))

        self.position = np.array([x, y])            
        self.visited_cells.add((x, y))
        if not self.env.fire_extinguished: self.fire_found = self.observe() # Observe for free after every action
        self.history.append(self.state)     
        self.time += 1
        self.steps_since_last_comm += 1

        return telemetry_packet


    def observe(self):
        """Update belief based on observation"""
        x_check = (self.x - self.window_size // 2 <= self.env.fire_pos[0] <= self.x + self.window_size // 2)
        y_check = (self.y - self.window_size // 2 <= self.env.fire_pos[1] <= self.y + self.window_size // 2)
        fire_observed = x_check and y_check
        
        self.belief_state.update_from_observation(self.position, self.window_size, fire_observed)        
        
        if fire_observed: 
            print(f"Drone {self.drone_id} found fire at position {self.env.fire_pos}!")
            
            if np.array_equal(self.position, self.env.fire_pos):
                self.action(6) # Extinguish the fire

            else: # Move towards the fire
                if self.x < self.env.fire_pos[0]:
                    self.action(4)
                elif self.x > self.env.fire_pos[0]:
                    self.action(3)
                elif self.y < self.env.fire_pos[1]:
                    self.action(1)
                elif self.y > self.env.fire_pos[1]:
                    self.action(2)

        return fire_observed


    def create_telemetry_packet(self):
        """Creates telemetry packet with belief state"""
        packet = {
            'sender_id': self.drone_id,
            'timestamp': self.time,
            'position': self.position.copy(),
            'belief_state': deepcopy(self.belief_state),
            'visited_cells': self.visited_cells.copy(),
            'history': deepcopy(self.history)
        }
        return packet


    def receive_telemetry(self, packet, communication_noise=0.1):
        """Receive and merge belief states"""
        other_visited = packet['visited_cells']
        self.visited_cells.update(other_visited)
        if 'belief_state' in packet:
            self.belief_state.merge(packet['belief_state'])

    def _get_best_value(self, belief, position, visited, depth):
        """
        Recursive helper to calculate the best Q-value from a given state with limited lookahead.
        """
        if depth == 0:
            return 0.0
            
        max_q = -float('inf')
        moves = { 1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}
        current_entropy = belief.get_entropy()
        
        for _, (dx, dy) in moves.items():
            nx = max(0, min(self.env.grid_size - 1, position[0] + dx))
            ny = max(0, min(self.env.grid_size - 1, position[1] + dy))
            
            # R(b, a)
            bonus = self.exploration_bonus if (nx, ny) not in visited else -self.exploration_bonus
            reward_action = bonus - self.movement_cost - self.time_cost
            
            # P(o|b, a)
            half = self.window_size // 2
            x_min, x_max = max(0, nx - half), min(self.env.grid_size, nx + half + 1)
            y_min, y_max = max(0, ny - half), min(self.env.grid_size, ny + half + 1)
            
            prob_see_fire = np.sum(belief.belief_grid[x_min:x_max, y_min:y_max])
            prob_see_nothing = 1.0 - prob_see_fire
            
            # U(b'| o = Fire) - Immediate high value (entropy reduction)
            gain_see_fire = current_entropy
            
            # U(b'| o = Nothing)
            temp_belief = deepcopy(belief)
            temp_belief.update_from_observation((nx, ny), self.window_size, fire_found=False)
            gain_see_nothing = current_entropy - temp_belief.get_entropy()
            
            # Recursive Value
            future_val = 0.0
            if depth > 1 and prob_see_nothing > 0:
                new_visited = visited.copy()
                new_visited.add((nx, ny))
                future_val = self._get_best_value(temp_belief, (nx, ny), new_visited, depth - 1)
            
            # Q(b, a)
            val_nothing = gain_see_nothing + self.gamma * future_val
            q = reward_action + self.gamma * (prob_see_fire * gain_see_fire + prob_see_nothing * val_nothing)
            
            if q > max_q:
                max_q = q
                
        return max_q

    def decide_action_pomdp(self):
        """
        Lookahead POMDP Planning (Depth=2).
        Returns the action index (1-5) with the highest Q-value.
        """
        best_action = 0 
        max_q_value = -float('inf')
        lookahead_depth = 2
        
        current_entropy = self.belief_state.get_entropy()
        
        # Actions: 0=Stay 1=Up, 2=Down, 3=Left, 4=Right
        moves = { 1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}

        # --- Check the Q-value for performing a movement ---
        for action_idx, (dx, dy) in moves.items():
            nx = max(0, min(self.env.grid_size - 1, self.x + dx))
            ny = max(0, min(self.env.grid_size - 1, self.y + dy))
            
            # --- R(b, a) ---
            bonus = 0.0
            if (nx, ny) not in self.visited_cells:
                bonus = self.exploration_bonus # Reward visiting new cells
            else: 
                bonus = -self.exploration_bonus # Penalize revisiting cells

            reward_action = bonus - self.movement_cost - self.time_cost
            
            # --- P(o|b, a) ---
            half = self.window_size // 2
            x_min, x_max = max(0, nx - half), min(self.env.grid_size, nx + half + 1)
            y_min, y_max = max(0, ny - half), min(self.env.grid_size, ny + half + 1)
            
            # P(o = Fire)
            prob_see_fire = np.sum(self.belief_state.belief_grid[x_min:x_max, y_min:y_max])
            
            # P(o = Nothing)
            prob_see_nothing = 1.0 - prob_see_fire
            
            # U(b'| o = Fire)
            gain_see_fire = current_entropy

            # U(b'| o = Nothing)
            temp_belief = deepcopy(self.belief_state)
            temp_belief.update_from_observation((nx, ny), self.window_size, fire_found=False)
            gain_see_nothing = current_entropy - temp_belief.get_entropy()
            
            # --- Future Value (Lookahead) ---
            future_val = 0.0
            if lookahead_depth > 1 and prob_see_nothing > 0:
                new_visited = self.visited_cells.copy()
                new_visited.add((nx, ny))
                future_val = self._get_best_value(temp_belief, (nx, ny), new_visited, lookahead_depth - 1)

            # Q(b, a) = R(b, a) + \gamma*\sum(P(o|b, a)*U(b))
            val_nothing = gain_see_nothing + self.gamma * future_val
            q_value = reward_action + self.gamma*(prob_see_fire*gain_see_fire + prob_see_nothing*val_nothing)
            
            print(f"Time step {self.time} | Action {action_idx} | Reward: {reward_action:.2f} | Q-Value: {q_value:.2f}")
            print(f"\tP(o = Fire):    {prob_see_fire:.6f}\tU(b'| o = Fire):    {gain_see_fire:.4f}")
            print(f"\tP(o = Nothing): {prob_see_nothing:.6f}\tU(b'| o = Nothing): {val_nothing:.4f} (Inc. Future: {future_val:.2f})")
            
            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action_idx

        # --- Check the Q-value for performing communication ---
        # Heuristic: Communicate if we have gathered significant information (Low Entropy relative to prior)
    
        # R(b, a = communicate)
        reward_comm = -self.comm_cost - self.time_cost

        # U(b' | a = communicate) - Value of sharing our knowledge
        # We use (Max Entropy - Current Entropy) as a proxy for "Information Possessed"
        max_entropy = np.log(self.env.grid_size * self.env.grid_size)
        information_possessed = max(0.0, max_entropy - current_entropy)
        estimated_comm_gain = information_possessed * (1.0 + 0.4 * self.steps_since_last_comm)

        # Q(b, a = communicate) = R(b, a = communicate) + U(b' | a = communicate)
        q_comm = reward_comm + estimated_comm_gain 
        
        print(f"(Q_comm: {q_comm:.2f} vs Q_move: {max_q_value:.2f})")

        if q_comm > max_q_value:
            print(f"Drone {self.drone_id} decided to COMMUNICATE.")
            return 5
            
        return best_action