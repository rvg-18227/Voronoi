"""Player module for Group 8 - Voronoi."""
from distutils.spawn import spawn
import logging
import math
import os
import pickle
from typing import List, Tuple
from xmlrpc.client import Boolean

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import transform
import simplekml
import sympy
from sympy import Circle  # , Point


class Player:
    """A class to represent a player in a 4-person battlefield."""

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
            precomp_dir: str,

    ) -> None:
        """Initialise the player with given skill.

        Args:
            rng (np.random.Generator): numpy random number generator, use this
                                       for same player behavior across run
            logger (logging.Logger): logger use this like logger.info("msg")
            total_days (int): total number of days, the game is played
            spawn_days (int): number of days after which new units spawn
            player_idx (int): index used to identify the player among the four
                              possible players
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
        self.is_stay_guard = False
        self.guard_list = []
        self.choose_guard = False
        self.enemy_distance = 0 ## how far ahead to look for enemy units before moving forward

    def play(
            self,
            unit_id,
            unit_pos,
            map_states,
            current_scores,
            total_scores
    ) -> List[Tuple[float, float]]:
        """Function which based on current game state returns the distance and
           angle of each unit active on the board.

        Args:
            unit_id (list(list(str))): contains the ids of each player's units
                                                (unit_id[player_idx][x])
            unit_pos (list(list(float))): contains the position of each unit
                                          currently present on the map
                                                (unit_pos[player_idx][x])
            map_states (list(list(int)): contains the state of each cell, using
                                         the x, y coordinate system
                                                (map_states[x][y])
            current_scores (list(int)): contains the number of cells currently
                                        occupied by each player
                                                (current_scores[player_idx])
            total_scores (list(int)): contains the cumulative scores up until
                                      the current day (total_scores[player_idx]

        Returns:
            List[Tuple[float, float]]: Return a list of tuples consisting of
                                       distance and angle in radians to move
                                       each unit of the player
        """

        moves = []

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

        # current_days/total_days = current_points/total points

        points = unit_pos[self.player_idx]
        base_point = points[0]
        self.total_points = self.total_days//self.spawn_days
        self.current_day = (len(points)/(self.total_days //
                            self.spawn_days) * self.total_days)//1  # rough estimate
        print(self.current_day)
        min_distance = 0.5

        f = 3
        time = self.total_days//self.spawn_days
        radius = math.sqrt((f * self.max_dim ** 2 * 4 / math.pi))
        max_distance = math.pi * radius / 2 * time
        # if len(points) == 1:  # the day when we first spawn dont move
        #     distance = sympy.Min(1, 0)
        #     angle = sympy.atan2(0, 1)
        #     moves.append((distance, angle))
        # elif len(points) == 2:
        #     # we have two units now!

        #     time = self.total_days//self.spawn_days
        #     # in this case 3 is just we are taking 1/3 of the area
        #     radius = self.spawn_days * (self.max_dim ** 2) * \
        #         f / (6 * time * math.pi ** 2)-0.5
        #     # move each troop outward in the form of a circle?
        #     distance = sympy.Min(radius)
        #     angle1 = sympy.atan2(100 - points[0].y,
        #                          100 - points[0].x)
        #     angle2 = sympy.atan2(100 - points[1].y,
        #                          100 - points[1].x)
        #     moves.append((distance, angle1))
        #     moves.append((distance, angle2))
        # else:
        # start spreading to other places

        newest_point = points[-1]
        p_new, p_base = Point(newest_point), Point(base_point)
        current_radius = 0
        if len(points) > 1:
            point1 = points[1]
            p1 = Point(point1)
            current_radius = p_base.distance(p1)

            # new point spanwed!!! time to spread :)
            current_radius += 1
            # some code to spread
        moves = self.spread_points(current_radius, points)
        #point_dist_list = []
        # for i, item in enumerate(points):
        #     if i == 0:
        #         continue
        #     p1, p2 = points[i - 1], points[i]
        #     point_dist_list.append(p1.distance(p2))
        # if (min(point_dist_list) <= min_distance
        #         or max(point_dist_list) >= max_distance):
        #moves = self.spread_points(current_radius, points)
        # else:
        #     # dont move
        #     distance = sympy.Min(1, 0)
        #     angle = sympy.atan2(0, 1)
        #     moves.append((distance, angle))

        if self.current_day >= 40 and self.is_stay_guard == False:
            if self.choose_guard == False:
                # the three guards as index in the points
                self.guard_list = [len(points)-1, len(points)-2, len(points)-3]
                self.choose_guard = True
            moves = self.move_stay_guard(points, moves)
        for i in range(len(moves)):
            moves[i] = self.transform_move(moves[i])
            print("in transform")       
        return moves

    def spread_points(
        self,
        radius: float,
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
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
        size = len(points)
        index = 0
        # variable base on the number of points that will be geenrated total
        angle_jump = size/self.total_points*10
        angle_start = 45
        for item in points:
            index += 1
            distance = 1
            angle = (((index) * (angle_jump) + angle_start)) % 90
            if index in self.guard_list:
                distance = 0
            moves.append((distance, angle*(math.pi / 180)))

        return moves

    def move_stay_guard(
        self,
        points: List[Tuple[float, float]],
        moves: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        # move the last three points to guard the base
        # with the coordinate (1,0); (1,1) : (0,1)
        # remove the last three points
        guard_moves = []
        is_guard = []
        angles = [0, 45, 90]
        for i in range(len(self.guard_list)):
            guard = points[i]
            moves.pop(i)
            guard_point = Point(guard)
            g_s_dist = guard_point.distance(self.spawn_point)
            g_s_ang = self.angle_between(guard_point, self.spawn_point)
            if g_s_dist == 1 and g_s_ang == angel[i]:
                guard_moves.append((0, 0))
                is_guard.append(0)
            else:
                dist = min(g_s_dist, 1)
                angel = g_s_ang
                guard_moves.append((dist, angel))
        # move the points back to the base so that the coordinates would the right
        
        if sum(is_guard) == 0:
            self.is_stay_guard = True
        moves += guard_moves
        return moves

    def angle_between(self, p1: Point, p2 : Point) -> float:
        p1 = np.array(p1)
        p2 = np.array(p2)
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))
    def transform_move (self, dist_ang: Tuple[float, float]) -> Tuple[float, float]:
        dist, rad_ang = dist_ang
        return (dist, rad_ang - (math.pi/2 * self.player_idx))
    

    def is_safe(self) ->  Boolean:
        #TODO the safety heuristic
        return False