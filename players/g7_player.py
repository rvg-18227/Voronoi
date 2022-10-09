import os
import pickle
import numpy as np
import sympy
import logging
from typing import Tuple
from math import pi, dist
import numpy as np


class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger, total_days: int, spawn_days: int,
                 player_idx: int, spawn_point: sympy.geometry.Point2D, min_dim: int, max_dim: int, precomp_dir: str) \
            -> None:
        """Initialise the player with given skill.

            Args:
                rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
                logger (logging.Logger): logger use this like logger.info("message")
                total_days (int): total number of days, the game is played
                spawn_days (int): number of days after which new units spawn
                player_idx (int): index used to identify the player among the four possible players
                spawn_point (sympy.geometry.Point2D): Homebase of the player
                min_dim (int): Minimum boundary of the square map
                max_dim (int): Maximum boundary of the square map
                precomp_dir (str): Directory path to store/load pre-computation
        """

        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))

        # # precompute check
        # if os.path.isfile(precomp_path):
        #     # Getting back the objects:
        #     with open(precomp_path, "rb") as f:
        #         self.obj0, self.obj1, self.obj2 = pickle.load(f)
        # else:
        #     # Compute objects to store
        #     self.obj0, self.obj1, self.obj2 = _

        #     # Dump the objects
        #     with open(precomp_path, 'wb') as f:
        #         pickle.dump([self.obj0, self.obj1, self.obj2], f)

        # Default variables passed into function
        self.rng = rng
        self.logger = logger
        self.spawn_days = spawn_days
        self.total_days = total_days
        self.spawn_point = [spawn_point.x, spawn_point.y]
        self.player_idx = player_idx
        self.min_dim = min_dim
        self.max_dim = max_dim

        # Experimenal variables
        self.other_players = [0,1,2,3]
        self.other_players.remove(self.player_idx)
        self.last_n_days_positions = {}
        self.hostility_registry = {}

        for other_player in self.other_players:
            self.last_n_days_positions[other_player] = []
            self.hostility_registry[other_player] = 1

        self.n = 5
        self.day = 0


    def basic_aggressiveness(self, prev_position, current_position) -> float:
        """Function which based on current and past posisitions returns the difference in avg distance from spawn point.
            Is fairly inaccurate due to summing all units distance when one or two units could cause lots of damage in a sparse game

                Args:
                    unit_id (list(list(str))): contains the position of each unit on one team at some previous time
                    current_position (list(list(float))): contains the position of each unit on one team currently
                Returns:
                    float: Where greater values indicate more aggressive players
                """
        total_prev = 0
        total_current = 0
        for p in prev_position:
            total_prev = abs(dist(self.spawn_point, [p.x, p.y]))

        for p in current_position:
            total_current = abs(dist(self.spawn_point, [p.x, p.y]))

        return total_current/len(current_position) - total_prev/len(prev_position)
        

    def play(self, unit_id, unit_pos, map_states, current_scores, total_scores) -> [tuple[float, float]]:
        """Function which based on current game state returns the distance and angle of each unit active on the board

                Args:
                    unit_id (list(list(str))): contains the ids of each player's units (unit_id[player_idx][x])
                    unit_pos (list(list(float))): contains the position of each unit currently present on the map
                                                    (unit_pos[player_idx][x])
                    map_states (list(list(int)): contains the state of each cell, using the x, y coordinate system
                                                    (map_states[x][y])
                    current_scores (list(int)): contains the number of cells currently occupied by each player
                                                    (current_scores[player_idx])
                    total_scores (list(int)): contains the cumulative scores up until the current day
                                                    (total_scores[player_idx]

                Returns:
                    List[Tuple[float, float]]: Return a list of tuples consisting of distance and angle in radians to
                                                move each unit of the player
                """

        moves = []
        locx = 0
        locy = 0
        if self.player_idx == 0:
            locx = 0
            locy = 0
        elif self.player_idx == 1:
            locx = 0
            locy = 75
        elif self.player_idx == 2:
            locx = 75
            locy = 75
        elif self.player_idx == 3:
            locx = 75
            locy = 0
            
        attackers = []
            
        for i in range(25-1):
            for j in range(25-1):
                if map_states[i + locx][j + locy] != self.player_idx+1 and map_states[i + locx][j + locy] > 0 and map_states[i + locx][j + locy] not in attackers:
                    attackers.append(map_states[i + locx][j + locy])
        
        if self.player_idx == 0: # Top left orange
            if len(attackers) != 0:
                for i in attackers:
                    if len(attackers) > 1:
                        angles = [pi/2, pi/2, pi/4, 2*pi, 2*pi]
                    if i == 2:
                        angles = [pi/2, pi/2, pi/2, pi/2, pi/2]
                    elif i == 4:
                        angles = [2*pi, 2*pi, 2*pi, 2*pi, 2*pi]
            else:
                angles = [(2*pi/8), (4*pi/8), 0, (3*pi/8), (pi/8) ]
        elif self.player_idx == 1: # Bottom left pink
            if len(attackers) != 0:
                for i in attackers:
                    if len(attackers) > 1:
                        angles = [2*pi, 2*pi, (7*pi)/4, (3*pi)/2, (3*pi)/2]
                    if i == 1:
                        angles = [2*pi, 2*pi, 2*pi, 2*pi, 2*pi]
                    elif i == 3:
                        angles = [(3*pi)/2, (3*pi)/2, (3*pi)/2, (3*pi)/2, (3*pi)/2]
            else:
                angles = [7*pi/4, 3*pi/2, 0, 13*pi/8, 15*pi/8] 
        elif self.player_idx == 2: # Bottom right blue
            if len(attackers) != 0:
                for i in attackers:
                    if len(attackers) > 1:
                        angles = [(3*pi)/2, (3*pi)/2, (5*pi)/4 , pi, pi]
                    if i == 4:
                        angles = [(3*pi)/2, (3*pi)/2, (3*pi)/2, (3*pi)/2, (3*pi)/2]
                    elif i == 2:
                        angles = [pi, pi, pi, pi, pi]
            else:
                angles = [5*pi/4, 3*pi/2, pi, 11*pi/8, 9*pi/8] 
        else: # Top right yellow
            if len(attackers) != 0:
                for i in attackers:
                    if len(attackers) > 1:
                        angles = [pi, pi, (3*pi)/4 , pi/2, pi/2]
                    if i == 1:
                        angles = [pi, pi, pi, pi, pi]
                    elif i == 3:
                        angles = [pi/2, pi/2, pi/2, pi/2, pi/2]
            else:
                angles = [3*pi/4, pi, pi/2, 7*pi/8, 5*pi/8]
        
        total_units = len(unit_id[self.player_idx])

        # Find aggressivness of other players maybe use this to adjust angles of units to be defensive
        
        for other_player in self.other_players:
            if len(unit_pos[other_player]) != 0:
                if len(self.last_n_days_positions[other_player]) == self.n:
                    change = self.basic_aggressiveness(self.last_n_days_positions[other_player].pop(0), unit_pos[other_player])
                    self.hostility_registry[other_player] = change
                self.last_n_days_positions[other_player].append(unit_pos[other_player])

        for i in range(total_units):
            moves.append((1, angles[i % len(angles)]))

        return moves
