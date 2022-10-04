import os
import pickle
import numpy as np
import sympy
import logging
from typing import Tuple
from sympy.geometry import Point2D

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

        self.rng = rng
        self.logger = logger

        self.player_idx = player_idx

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
        self.map_states = map_states
        moves = []
        for unit in unit_pos[self.player_idx]:
            move = self.transform_move(np.arctan2(1, 1))
            if self.check_move(unit, move) == self.player_idx:
                moves.append(self.transform_move(0, 0))
            moves.append(move)
        return moves

    def check_move(self, unit_pos, move) -> int:
        """Checks if the move is valid and returns the player index of the unit occupying the cell
                Args:
                    unit_pos (Tuple(float, float)): current position of the unit
                    move (Tuple(float, float)): move to be made as (distance, angle)

                Returns:
                    int: player index of the player who owns the cell
        """
        distance, angle = move
        end_pos = (unit_pos[0] + distance * np.cos(angle), unit_pos[1] + distance * np.sin(angle))
        return self.map_states[np.floor(end_pos[0])][np.floor(end_pos[1])]


    def transform_move(self, angle: float, distance=1) -> tuple[float, float]:
        """Transforms the distance and angle to the correct format for the game engine
                Args:
                    angle (float): angle in radians
                    distance (float): distance to move
                Returns
                    Tuple[float, float]: distance and angle in correct format
        """
        if self.player_idx == 0:
            return (distance, angle)
        elif self.player_idx == 1:
            return (distance, angle - np.pi/2)
        elif self.player_idx == 2:
            return (distance, angle + np.pi)
        else:# self.player_idx == 3:
            return (distance, angle + np.pi / 2)
