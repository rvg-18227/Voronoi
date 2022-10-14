"""Player module for Group 8 - Voronoi."""
from distutils.spawn import spawn
import logging
import math
import os
import pickle
from re import L
from typing import List, Tuple
from xmlrpc.client import Boolean

from traitlets import Float

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

        self.enemy_distance = 0 ## how far ahead to look for enemy units before moving forward
        angels = [0,45,90]
        self.angels = []
        for angel in angels:
             self.angels.append(angel  - (math.pi/2 * self.player_idx))
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
        guard_num = 3 # number of guards to protect the base
        moves = []
        points = unit_pos[self.player_idx]
        ids = unit_id[self.player_idx]
        base_point = points[0]
        self.total_points = self.total_days//self.spawn_days
        self.current_day = (len(points)/(self.total_days //
                            self.spawn_days) * self.total_days)//1  # rough estimate
        min_distance = 0.5
        self.make_point_dict(points,ids) ## intialize the look up dict for id => points
        f = 3
        time = self.total_days//self.spawn_days
        radius = math.sqrt((f * self.max_dim ** 2 * 4 / math.pi))
        max_distance = math.pi * radius / 2 * time
        newest_point = points[-1]
        p_new, p_base = Point(newest_point), Point(base_point)
        current_radius = 0
        print(map_states[50])
        if len(points) > 1:
            point1 = points[1]
            p1 = Point(point1)
            current_radius = p_base.distance(p1)

            # new point spanwed!!! time to spread :)
            current_radius += 1
            # some code to spread
        new_guard = []
        for i in range(len(self.guard_list)):
                guard = self.guard_list[i]
                if guard  in ids:
                    new_guard.append(guard)
        self.guard_list = new_guard
        if self.current_day >= 40 and len(self.guard_list) <guard_num:
            # the three guards as index in the points
            #grab the last three id and insert them into the list 
            cur_guard = len(ids)-1
            while len(self.guard_list) < guard_num and cur_guard > -1:
                if ids[cur_guard] not in self.guard_list:
                    self.guard_list.append( ids[cur_guard])
                cur_guard -=1
            

        moves = self.spread_points(current_radius, points)
        for i in range(len(moves)):
            moves[i] = self.transform_move(moves[i])
        return moves







    def make_point_dict(self,
    units:List[Tuple[float, float]]
    ,ids:List[int]):
    ## creates the look up dictionary for id and unit location
        point_dict = {}
        for i in range(len(ids)):
            point_dict[ids[i]] = units[i]
        self.point_dict = point_dict

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
        guard_index = 0
        guard_dict = {}
        for guard in self.guard_list:
            guard_dict[guard]= self.point_dict[guard]
        for i in points:
            index += 1
            distance = 1
            angle = (((index) * (angle_jump) + angle_start)) % 90
            if i in guard_dict.items() and self.is_stay_guard == False: ## call the move guard function
                distance,angle = self.move_stay_guard(i,self.angles[guard_index])
                guard_index += 1
            moves.append((distance, angle*(math.pi / 180)))

        return moves

    def move_stay_guard(
        self,
        guard_point: Point,
        angle : Float
    ) -> List[Tuple[float, float]]:
        # move the last three points to guard the base
        # with the coordinate (1,0); (1,1) : (0,1)
        # remove the last three points
        move = []
        is_guard = []
        g_s_dist = math.abs(guard_point.distance(self.spawn_point))
        g_s_ang = abs(angle-self.angle_between(guard_point, self.spawn_point))
        if g_s_dist == 1 and g_s_ang == angle:
                move.append((0, 0))
                is_guard.append(0)
        else:
                dist = min(g_s_dist, 1)
                angel = g_s_ang
                move.append((dist, angel))
        # move the points back to the base so that the coordinates would the right
        return move

    def angle_between(self, p1: Point, p2 : Point) -> float:
        p1 = np.array(p1)
        p2 = np.array(p2)
        dy = p1[1]-p2[1]
        dx = p1[0]-p2[0]
        angel = math.atan2(dy, dx)
        return angel
    def transform_move (self, dist_ang: Tuple[float, float]) -> Tuple[float, float]:
        dist, rad_ang = dist_ang
        return (dist, rad_ang - (math.pi/2 * self.player_idx))
    

    def is_safe(self) ->  Boolean:
        #TODO the safety heuristic
        return False