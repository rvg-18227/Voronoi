import os
import pickle
import numpy as np
import sympy
import logging
from typing import Tuple, List
import math
from shapely.geometry import Point


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

    def transform_move (self, dist_ang: Tuple[float, float]) -> Tuple[float, float]:
        dist, rad_ang = dist_ang
        return (dist, rad_ang - (math.pi/2 * self.player_idx))

    def get_home_coords(self):
        if self.player_idx == 0:
            return Point(0.5, 0.5)
        elif self.player_idx == 1:
            return Point(0.5, 99.5)
        elif self.player_idx == 2:
            return Point(99.5, 99.5)
        elif self.player_idx == 3:
            return Point(99.5, 0.5)

    def get_wall_dist(self, current_point):
        current_x = current_point.x
        current_y = current_point.y
        dist_to_top = Point(current_x, 0).distance(current_point)
        dist_to_bottom = Point(current_x, 100).distance(current_point)
        dist_to_right = Point(0, current_y).distance(current_point)
        dist_to_left = Point(100, current_y).distance(current_point)
        return {"top": dist_to_top, "bottom": dist_to_bottom, "right": dist_to_right, "left": dist_to_left}

    def get_closest_friend(self, current_unit, current_pos, unit_pos, unit_id):
        closest_unit_dist = math.inf
        closest_unit = math.inf
        for i in range(len(unit_pos[self.player_idx])):
            if i == current_unit:
                continue
            friend_unit = unit_id[self.player_idx][i]
            friend_unit_pos = unit_pos[self.player_idx][i]
            dist = friend_unit_pos.distance(current_pos)
            if dist < closest_unit_dist:
                closest_unit_dist = dist
                closest_unit = friend_unit
        return {"unit_id": closest_unit, "distance": closest_unit_dist}

    def get_forces(self, unit_id, unit_pos):
        forces = {}
        for i in range(len(unit_id[self.player_idx])):
            unit = unit_id[self.player_idx][i]
            current_pos = unit_pos[self.player_idx][i]
            forces[unit] = {}
            home_coord = self.get_home_coords()
            forces[unit]["dist_home"] = home_coord.distance(current_pos)
            forces[unit]["dist_walls"] = self.get_wall_dist(current_pos)
            if len(unit_id[self.player_idx]) > 1:
                closest_friend = self.get_closest_friend(i, current_pos, unit_pos, unit_id)
                forces[unit]["dist_friend"] = closest_friend
            else:
                forces[unit]["dist_friend"] = "None"
        return forces


    def play(self, unit_id, unit_pos, map_states, current_scores, total_scores) -> List[Tuple[float, float]]:
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
        angle_jump = 10
        angle_start = 45
        forces = self.get_forces(unit_id, unit_pos)

        for i in range(len(unit_id[self.player_idx])):
            distance = 1

            angle = (((i) * (angle_jump) + angle_start ))%90

            moves.append((distance, angle* (math.pi / 180)))

        return [self.transform_move(move) for move in moves]
        #return moves
