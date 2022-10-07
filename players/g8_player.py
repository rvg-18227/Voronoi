import logging
import math
import os
import pickle
from typing import List, Tuple

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import transform
import simplekml
import sympy
from sympy import Circle  # ,Point


class Player:
    def __init__(
            self,
            rng: np.random.Generator,
            logger: logging.Logger,
            total_days: int,
            spawn_days: int,
            player_idx: int,
            spawn_point: sympy.geometry.Point2D,
            min_dim: int,
            max_dim: int,
            precomp_dir: str) -> None:
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
        self.total_days = total_days
        self.spawn_days = spawn_days
        self.player_idx = player_idx
        self.spawn_point = spawn_point
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.precomp_dir = precomp_dir

    def play(
            self,
            unit_id,
            unit_pos,
            map_states,
            current_scores,
            total_scores) -> List[Tuple[float, float]]:
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

        # distance = sympy.Min(1, 100 - unit_pos.x)
        # this is right for us
        # angle = sympy.atan2(100 - unit_pos.y, 100 - unit_pos.x)
        # moves.append((distance, angle))
        # calculate the distance between each unit, see if its within the acceptable range
        # p1.distance(p2) would return the distance between two points
        # if distance is within a set range ( due to roudning error), do nothing
        # if too big, shrink
        # if too small, expand

        # for i in range(len(unit_id[self.player_idx])):
        #     if self.player_idx == 0:
        #         distance = sympy.Min(1, 100 - unit_pos[self.player_idx][i].x)
        #         angle = sympy.atan2(100 - unit_pos[self.player_idx][i].y,
        #                             100 - unit_pos[self.player_idx][i].x)
        #         moves.append((distance, angle))
#            elif self.player_idx == 1:
#                distance = sympy.Min(1, 100 - unit_pos[self.player_idx][i].x)
#                angle = sympy.atan2(0.5 - unit_pos[self.player_idx][i].y,
#                                    0.5 - unit_pos[self.player_idx][i].x)
#                moves.append((distance, angle))
#            elif self.player_idx == 2:
#                distance = sympy.Min(1, self.rng.random())
#                angle = sympy.atan2(-self.rng.random(), -self.rng.random())
#                moves.append((distance, angle))
#            else:
#                distance = sympy.Min(1, 0)
#                angle = sympy.atan2(0, 1)
#                moves.append((distance, angle))

        # return moves

        points = unit_pos[self.player_idx]
        base_point = points[0]
        min_distance = 0.5
        f = 3
        t = self.total_days//self.spawn_days
        r = (f * self.max_dim ** 2 * 4 / math.pi)**(0.5)
        max_distance = math.pi * r / 2 * t
        if len(points) == 1:  # the day when we first spawn dont move
            distance = sympy.Min(1, 0)
            angle = sympy.atan2(0, 1)
            moves.append((distance, angle))
        elif len(points) == 2:
            # we have two units now!

            t = self.total_days//self.spawn_days
            # in this case 3 is just we are taking 1/3 of the area
            r = self.spawn_days * (self.max_dim ** 2) * \
                f / (6 * t * math.pi ** 2)-0.5
            # move each troop outward in the form of a circle?
            distance = sympy.Min(r)
            angle1 = sympy.atan2(100 - points[0].y, 100 -
                                 points[0].x)
            angle2 = sympy.atan2(100 - points[1].y, 100 -
                                 points[1].x)
            moves.append((distance, angle1))
            moves.append((distance, angle2))
        else:
            # start spreading to other places

            newest_point = points[-1]
            p_n, p_b = Point(newest_point), Point(base_point)
            point1 = points[1]
            p1 = Point(point1)
            current_radius = p_b.distance(p1)
            if p_n.distance(p_b) == 0:
                # new point spanwed!!! time to spread :)
                current_radius += 1
                # some code to spread
                moves = self.spread_points(self, current_radius, points)
            step = 1
            point_dist_list = []
            for i, item in enumerate(points):
                if i == 0:
                    continue
                p1, p2 = points[i - 1], points[i]
                point_dist_list.append(p1.distance(p2))
            if (min(point_dist_list) <= min_distance
                    or max(point_dist_list) >= max_distance):
                moves = self.spread_points(current_radius, points)
            else:
                # dont move
                distance = sympy.Min(1, 0)
                angle = sympy.atan2(0, 1)
                moves.append((distance, angle))

        return moves

    def spread_points(self, radius: float, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Get the spread points.

        Args:
            radius (float): The radius to be used for circular spread
            points [tuple[float, float]]: List of points for the spread

        Returns:

        """
        # how each point should move so that the points are evenly spread on
        # the edge of the circle
        moves = []
        # move each troop outward in the form of a circle?
        for item in points:
            distance = sympy.Min(1, 100 - item.x)
            angle = sympy.atan2(100 - item.y,
                                100 - item.x)  # this is right for us
            moves.append((distance, angle))

        return moves
