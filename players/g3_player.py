import logging
import math
import os
import pickle
from typing import Tuple, List
import copy

import numpy as np
from shapely.geometry import Point
from sklearn.cluster import KMeans
import sympy

LOG_LEVEL = logging.DEBUG

WALL_DENSITY = 0.1
WALL_RATIO = 0
PRESSURE_HI = 1000
PRESSURE_LO = 100

SCOUT_ALLY_SCALE = 2.0
SCOUT_BORDER_REPULSION_SCALE = 1.0


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
        self.logger.setLevel(LOG_LEVEL)
        
        self.us = player_idx
        self.homebase = np.array(spawn_point)
        self.day_n = 0

        self.our_units = None
        self.enemy_units = None
        self.enemy_offsets = None
        self.map_states = None

        self.target_loc = []

        self.initial_radius = 35

        base_angles = get_base_angles(player_idx)
        outer_wall_angles = np.linspace(start=base_angles[0], stop=base_angles[1], num=(total_days // spawn_days))
        self.midsorted_outer_wall_angles = midsort(outer_wall_angles)

    def debug(self, *args):
        self.logger.info(" ".join(str(a) for a in args))

    def push(self, unit_pos: List[List[Point]]):
        unit_pos = np.array([shapely_pts_to_tuples(pts) for pts in unit_pos])
        allies = unit_pos[self.us]
        enemies = np.delete(unit_pos, self.us, 0).flatten()

        k = math.ceil(len(allies) / 4)
        kmeans = KMeans(n_clusters=k).fit(allies)

        repelling_forces = [repelling_force_sum(enemies, c) for c in kmeans.cluster_centers_]
        exceed_pressure_lo = [higher_than_lo(force) for force in repelling_forces]
        soldier_moves = [_push_radially(allies[i], self.homebase, exceed_lo=exceed_pressure_lo[cid]) for i, cid in enumerate(kmeans.labels_)]

        self.debug([int(np.linalg.norm(force)) for force in repelling_forces])

        return soldier_moves

    def play(self, unit_id: List[List[str]], unit_pos: List[List[Point]], map_states: List[List[int]], current_scores: List[int], total_scores: List[int]) -> List[Tuple[float, float]]:
        """Function which based on current game state returns the distance and angle of each unit active on the board

                Args:
                    unit_id (list(list(str))): contains the ids of each player's units (unit_id[player_idx][x])
                    unit_pos (list(list(shapely.geometry.Point))): contains the position of each unit currently present on the map
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

        self.day_n += 1

        self.map_states = map_states

        float_unit_pos = [shapely_pts_to_tuples(pts) for pts in unit_pos]
        self.enemy_offsets = np.array([len(unit_pos[i]) for i in range(4) if i != self.us])
        self.enemy_units = np.concatenate([float_unit_pos[i] for i in range(4) if i != self.us])
        self.our_units = np.array(float_unit_pos[self.us])

        # EARLY GAME: form a 2-layer wall
        if self.day_n <= self.initial_radius:
            while len(unit_id[self.us]) > len(self.target_loc):
                # add new target_locations
                self.target_loc.append(
                    self.order2coord([self.initial_radius, self.midsorted_outer_wall_angles[len(unit_id[self.us]) - 1]]))
        
            return get_moves(shapely_pts_to_tuples(unit_pos[self.us]), self.target_loc)
        
        # MID_GAME: adjust formation based on opponents' positions
        num_scouts = 3
        scouts = np.array(float_unit_pos[self.us][:num_scouts])

        # remove first n troops as scouts...
        unit_pos = copy.deepcopy(unit_pos)
        unit_pos[self.us] = unit_pos[self.us][num_scouts:]
        defense_moves = self.push(unit_pos)

        offense_moves = self.move_scouts(scouts, np.arange(num_scouts))

        return offense_moves + defense_moves


    def order2coord(self, order) -> tuple[float, float]:
        dist, angle = order
        x = self.homebase[0] - dist * math.sin(angle)
        y = self.homebase[1] + dist * math.cos(angle)
        return (x, y)

    def explore(self, scout_unit, ally_units):
        enemy_k = min(20, math.ceil(self.enemy_units.shape[0]/4))
        enemy_clusters = KMeans(n_clusters=enemy_k).fit(self.enemy_units)
        ally_k = min(6, math.ceil(ally_units.shape[0]/4))
        ally_clusters = KMeans(n_clusters=ally_k).fit(ally_units)
        force = exploration_force(scout_unit, ally_clusters.cluster_centers_, enemy_clusters.cluster_centers_) \
            + border_repulsion(scout_unit, xmax=len(self.map_states), ymax=len(self.map_states[0]))
        return np.array([1, np.arctan2(force[1], force[0])])

    def move_scouts(self, scout_units, scout_ids):
        scout_moves = np.zeros_like(scout_units, dtype=float)
        # safety check
        ally_units = np.delete(self.our_units, scout_ids, axis=0)
        ally_dist = ((scout_units.reshape(-1, 1, 2) - ally_units.reshape(1, -1, 2)) ** 2).sum(axis=2)
        min_ally_id = ally_dist.argmin(axis=1)
        min_enemy_dist = ((scout_units.reshape(-1, 1, 2) - self.enemy_units.reshape(1, -1, 2)) ** 2).sum(axis=2).argmin(axis=1)
        for i in range(scout_units.shape[0]):
            if ally_dist[i, min_ally_id[i]] >= min_enemy_dist[i]:
                # retreat
                to_x, to_y = ally_units[min_ally_id[i]] - scout_units[i]
                scout_moves[i] = np.array([1, np.arctan2(to_y, to_x)])
            else:
                # explore
                scout_moves[i] = self.explore(scout_units[i], ally_units)

        return scout_moves.tolist()


# -----------------------------------------------------------------------------
#   Strategies
# -----------------------------------------------------------------------------

def _push_radially(pt, homebase, exceed_lo=False):
    if exceed_lo:
        # stay where we are
        return (0., 0.)

    towards_x, towards_y = np.array(pt) - np.array(homebase)
    angle = np.arctan2(towards_y, towards_x)
    
    return (1, angle)

# -----------------------------------------------------------------------------
#   Force (NumPy)
# -----------------------------------------------------------------------------
def exploration_force(curr_pt, ally_pts, enemy_pts, ally_scale=SCOUT_ALLY_SCALE):
    ally_v = curr_pt - ally_pts
    ally_force = inverse_force(ally_v)
    enemy_v = curr_pt - enemy_pts
    enemy_force = inverse_force(enemy_v)
    return ally_scale * ally_force + enemy_force

def border_repulsion(curr_pt, xmax, ymax, scale=SCOUT_BORDER_REPULSION_SCALE):
    border_pts = np.array([[0, curr_pt[1]], [xmax, curr_pt[1]], [curr_pt[0], 0], [curr_pt[0], ymax]])
    border_v = curr_pt - border_pts
    return scale * inverse_force(border_v)

def inverse_force(v):
    mag = np.sqrt((v ** 2).sum(axis=1, keepdims=True))
    force = (v / mag / mag).sum(axis=0)
    return force
    
# -----------------------------------------------------------------------------
#   Force
# -----------------------------------------------------------------------------
# NOTE: The code below are referenced from Group 4

def force_vec(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[List[float], float]:
    v = np.array(p1) - np.array(p2)
    mag = np.linalg.norm(v)
    unit = v / mag
    return unit, mag

def repelling_force(p1: Tuple[float, float], p2: Tuple[float, float]) -> List[float]:
    dir, mag = force_vec(p1, p2)
    # Inverse magnitude: closer things apply greater force
    return dir * 1 / (mag)

def repelling_force_sum(pts: List[Tuple[float, float]], receiver: Tuple[float, float]) -> List[float]:
    return np.add.reduce([repelling_force(receiver, x) for x in pts])

def reactive_force(fvec: List[float]) -> List[float]:
    return fvec * (-1.)

def higher_than_lo(force):
    return True if np.linalg.norm(force) > PRESSURE_LO else False

# -----------------------------------------------------------------------------
#   Helper functions
# -----------------------------------------------------------------------------

def get_moves(unit_pos: List[Tuple[float, float]], target_loc: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    assert len(unit_pos) == len(target_loc), "get_moves: unit_pos and target_loc array length not the same"
    np_unit_pos = np.array(unit_pos, dtype=float)
    np_target_loc = np.array(target_loc, dtype=float)

    cord_diff = np_target_loc - np_unit_pos
    cord_diff_x = cord_diff[:, 0]
    cord_diff_y = cord_diff[:, 1]

    move_dist = np.linalg.norm(cord_diff, axis=1)
    move_dist[move_dist > 1] = 1.0
    move_angle = np.arctan2(cord_diff_y, cord_diff_x)
    
    move_arr = list(zip(move_dist, move_angle))
    return move_arr


def shapely_pts_to_tuples(points: List[Point]) -> List[Tuple[float, float]]:
    return list(map(shapely_pt_to_tuple, points))


def shapely_pt_to_tuple(point: Point) -> Tuple[float, float]:
    return ( float(point.x), float(point.y) )


def midsort(arr: List[float]) -> List[float]:
    n = len(arr)
    if n <= 2:
        return arr

    first_elem_added = False
    prev_midpoints = [0, n - 1]
    midsorted_arr = []

    while len(prev_midpoints) < n:
        curr_midpoints = []

        for i, (left_pt, right_pt) in enumerate(zip(prev_midpoints, prev_midpoints[1:])):
            mid_pt = (left_pt + right_pt) // 2

            if mid_pt != left_pt or mid_pt == 0:
                curr_midpoints.extend([left_pt, mid_pt])
                midsorted_arr.append(arr[mid_pt])

                if mid_pt == 0:
                    first_elem_added = True
            else:
                curr_midpoints.append(left_pt)

            if i == len(prev_midpoints) - 2:
                curr_midpoints.append(right_pt)

        prev_midpoints = curr_midpoints

    # add the LAST element in the original @arr
    if not first_elem_added:
        midsorted_arr.append(arr[0])
        
    midsorted_arr.append(arr[-1])

    return midsorted_arr


def get_base_angles(player_idx: int) -> Tuple[float, float]:
    """
    Returns the angles in radians of the two edges around player @player_idx's homebase.

    Example:

        The map of the voronoi game is a 100 * 100 grid. From the top left going clockwise
        are player 0, 1, 2, 3.

        Below is a visualization of player 3's base angles.
        
            (base angle + pi/2) 
                ^
                | 
                |     
                -------> (base angle)
                    -pi/2 - pi/2 * (player_index: 3)
              p3
    """
    base_angle = (-1) * ((math.pi / 2) + (player_idx * math.pi/2))
    return base_angle, base_angle + math.pi/2