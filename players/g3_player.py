import logging
import math
import os
import pickle
from typing import Tuple, List
import time

import numpy as np
from shapely.geometry import Point
from sklearn.cluster import KMeans


LOG_LEVEL = logging.DEBUG

WALL_DENSITY = 0.1
WALL_RATIO = 0
PRESSURE_HI_THRESHOLD = 3
PRESSURE_LO_THRESHOLD = 1.5
PRESSURE_LO, PRESSURE_MID, PRESSURE_HI = range(3) 

SCOUT_HOMEBASE_SCALE = 10.0
SCOUT_ENEMY_SCALE = 1
SCOUT_BORDER_SCALE = 5.0

COOL_DOWN = 15
CB_DURATION = 0  # days dedicated to border consolidation in each cycle
CB_START = 35    # the day to start the first cycle of border consolidation


class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger, total_days: int, spawn_days: int,
                 player_idx: int, spawn_point: Point, min_dim: int, max_dim: int, precomp_dir: str) \
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
        self.num_scouts = 3

        base_angles = get_base_angles(player_idx)
        outer_wall_angles = np.linspace(start=base_angles[0], stop=base_angles[1], num=(total_days // spawn_days))
        self.counter = 0
        self.midsorted_outer_wall_angles = midsort(outer_wall_angles)

        self.cb_scheduled = np.array([CB_START, CB_START + CB_DURATION])


    def debug(self, *args):
        self.logger.info(" ".join(str(a) for a in args))
    
    def get_radius(self, points):
        """Returns the radial distance of our soldier at @point to our homebase."""
        return np.sqrt(((points - self.homebase) ** 2).sum(axis=1))

    def push(self, scout_ids) -> List[Tuple[float, float]]:
        #allies = np.array(shapely_pts_to_tuples(unit_pos[self.us]))
        allies = np.delete(self.our_units, scout_ids, axis=0)  # not using slicing because scout_ids could be non-consecutive

        # enemies = [shapely_pts_to_tuples(troops) for i, troops in enumerate(unit_pos) if i != self.us]
        # flattened_enemies = np.concatenate((enemies[0], enemies[1], enemies[2]), axis=0)
        flattened_enemies = self.enemy_units

        k = math.ceil(len(allies) / 4)
        kmeans = KMeans(n_clusters=k).fit(allies)

        # ally_distances = np.array([self.get_radius(point) for point in kmeans.cluster_centers_])
        ally_distances = self.get_radius(kmeans.cluster_centers_)
        kmeans_radius = KMeans(n_clusters=min(3, k)).fit(ally_distances.reshape(-1, 1))

        max_cluster = kmeans_radius.labels_[0]

        #repelling_forces = [repelling_force_sum(flattened_enemies, c) for c in kmeans.cluster_centers_]
        repelling_forces = [exploration_force(c, flattened_enemies, ally_pts=None) for c in kmeans.cluster_centers_]
        pressure_levels = np.array([get_pressure_level(force) for force in repelling_forces])
        pressure_levels[np.array(kmeans_radius.labels_) != max_cluster] = PRESSURE_LO # index where point is not in outer radius
        soldier_moves = [self._push_radially(allies[i], plevel=pressure_levels[cid]) for i, cid in enumerate(kmeans.labels_)]

        self.debug(f'pressure: {[int(np.linalg.norm(force)) for force in repelling_forces]}')
        self.debug(f'moves: {soldier_moves}')

        return soldier_moves

    def _move_radially(self, pt: List[float], forward=True) -> Tuple[float, float]:
        """Moves @pt radially away from homebase, returns a tuple (distance, angle)
        to move the point.
        
        If @pt is at homebase, select an angle from self.midsorted_outer_wall_angles
        to move in.

        If @forward is True, move away from the homebase. Otherwise, towards the homebase.
        """
        direction = 1 if forward else -1

        if (pt == self.homebase).all():
            angle = self.midsorted_outer_wall_angles[self.counter % len(self.midsorted_outer_wall_angles)]
            self.counter += 1
        else:
            towards_x, towards_y = np.array(pt) - np.array(self.homebase)
            angle = np.arctan2(towards_y, towards_x)
        
        return (direction, angle)

    def _push_radially(self, pt: List[float], plevel=False) -> Tuple[float, float]:
        """Push, stay or retreat based on the pressure level, returns a tuple
        (distance, angle) to move the point.
        """
        if plevel == PRESSURE_LO:
            return self._move_radially(pt)
        elif plevel == PRESSURE_MID:
            # stay where we are
            return (0., 0.)
        else:
            return self._move_radially(pt, forward=False)

    def send_to_border(self, unit_id: List[str], soldiers: List[Point], map_states: List[List[int]]) -> List[Tuple[float, float]]:
        """Sends soldiers to consolidate our bolder."""
        return [(0., 0.)] * len(soldiers)

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

        self.map_states = np.array(map_states) - 1

        float_unit_pos = [shapely_pts_to_tuples(pts) for pts in unit_pos]
        self.enemy_offsets = np.array([len(unit_pos[i]) for i in range(4) if i != self.us])
        self.enemy_units = np.concatenate([float_unit_pos[i] for i in range(4) if i != self.us])
        self.our_units = np.array(float_unit_pos[self.us])

        # EARLY GAME: form a 2-layer wall
        if self.day_n <= self.initial_radius:
            self.debug(f'day {self.day_n}: form initial wall')

            while len(unit_id[self.us]) > len(self.target_loc):
                # add new target_locations
                self.target_loc.append(
                    self.order2coord([self.initial_radius, self.midsorted_outer_wall_angles[len(unit_id[self.us]) - 1]]))
        
            return get_moves(shapely_pts_to_tuples(unit_pos[self.us]), self.target_loc)
        elif self.day_n >= self.cb_scheduled[0] and self.day_n < self.cb_scheduled[1]:
            self.debug(f'day {self.day_n}: consoldiate border')

            if self.day_n == self.cb_scheduled[1] - 1:
                self.cb_scheduled += (COOL_DOWN + CB_DURATION)

            return self.send_to_border(unit_id[self.us], unit_pos[self.us], map_states)
        else:
            # MID_GAME: adjust formation based on opponents' positions
            self.debug(f'day {self.day_n}: cool down')

            scout_ids = np.arange(self.num_scouts)

            start = time.time()
            defense_moves = self.push(scout_ids)
            self.debug(f'Defense: {time.time()-start}s')
            
            start = time.time()
            offense_moves = self.move_scouts(scout_ids)
            self.debug(f'Offense: {time.time()-start}s')

            return offense_moves + defense_moves

    def order2coord(self, order: Tuple[float, float]) -> Tuple[float, float]:
        """Converts an order, tuple of (dist2homebase, angle), into a coordinate."""
        dist, angle = order
        x = self.homebase[0] + dist * math.cos(angle)
        y = self.homebase[1] + dist * math.sin(angle)
        return (x, y)

    def get_border(self):
        """Get border of our territory"""
        # trace along x axis to find the starting point
        if self.us < 2: # 0, 1
            for i in range(100):
                if self.map_states[i, 99*self.us] != self.us:
                    pt = (i-1, 99*self.us)
                    break
        else: # 2, 3
            for i in range(100):
                if self.map_states[99-i, 99*(3-self.us)] != self.us:
                    pt = (99-i+1, 99*(3-self.us))
                    break

        border = set()
        self._trace_border(pt, border)
        return np.array(list(border))

    def _trace_border(self, curr_pt, border_pts):
        """From a point, recurse through neighbors to find all border cells"""
        if curr_pt in border_pts:
            return
        else:
            border_pts.add(curr_pt)

        # check all 8 neighbors
        xmax, ymax = self.map_states.shape
        for x in range(3):
            for y in range(3):
                neighbor = (max(min(curr_pt[0]-1+x, xmax-1), 0), 
                            max(min(curr_pt[1]-1+y, ymax-1), 0))
                if neighbor != curr_pt and self._on_border((neighbor)):
                    self._trace_border(neighbor, border_pts)

    def _on_border(self, pt):
        """Check if given point is on the border"""
        # currently ignoring disputed cells
        if self.map_states[pt] != self.us: 
            return False

        xmax, ymax = self.map_states.shape
        neighbors = np.array([[min(pt[0]+1, xmax-1), pt[1]], 
            [max(pt[0]-1, 0), pt[1]],
            [pt[0], min(pt[1]+1, ymax-1)],
            [pt[0], max(pt[1]-1, 0)]])
        return any(self.map_states[neighbors[:, 0], neighbors[:, 1]] != self.us)

    def _explore(self, scout_unit, enemy_clusters, ally_clusters):
        homebase_force = inverse_force((scout_unit - self.homebase).reshape(1, 2))
        force = exploration_force(scout_unit, enemy_clusters, ally_pts=ally_clusters) \
            + SCOUT_BORDER_SCALE * border_repulsion(scout_unit, xmax=self.map_states.shape[0], ymax=self.map_states.shape[1]) \
            + SCOUT_HOMEBASE_SCALE * homebase_force
        return np.array([1, np.arctan2(force[1], force[0])])

    def _get_clusters(self, ally_units):
        # keep this incase we need it later
        # enemy_k = min(50, math.ceil(self.enemy_units.shape[0]/2))
        # enemy_clusters = KMeans(n_clusters=enemy_k).fit(self.enemy_units).cluster_centers_
        # ally_k = min(15, math.ceil(ally_units.shape[0]/2))
        # ally_clusters = KMeans(n_clusters=ally_k).fit(ally_units).cluster_centers_

        # change to random selection to speed up
        enemy_clusters = self.enemy_units[np.random.choice(np.arange(self.enemy_units.shape[0]), min(self.enemy_units.shape[0], 50), replace=False)]
        ally_clusters = ally_units[np.random.choice(np.arange(ally_units.shape[0]), min(ally_units.shape[0], 15), replace=False)]
        return enemy_clusters, ally_clusters

    def move_scouts(self, scout_ids):
        scout_units = self.our_units[scout_ids]
        scout_moves = np.zeros_like(scout_units, dtype=float)
        # safety check
        ally_units = np.delete(self.our_units, scout_ids, axis=0)
        ally_dist = ((scout_units.reshape(-1, 1, 2) - ally_units.reshape(1, -1, 2)) ** 2).sum(axis=2)
        min_ally_id = ally_dist.argmin(axis=1)
        min_enemy_dist = ((scout_units.reshape(-1, 1, 2) - self.enemy_units.reshape(1, -1, 2)) ** 2).sum(axis=2).min(axis=1)

        enemy_clusters, ally_clusters = self._get_clusters(ally_units)
        for i in range(scout_units.shape[0]):
            if ally_dist[i, min_ally_id[i]] >= min_enemy_dist[i] * 2:
                # retreat
                to_x, to_y = ally_units[min_ally_id[i]] - scout_units[i]
                scout_moves[i] = np.array([1, np.arctan2(to_y, to_x)])
            else:
                # explore
                scout_moves[i] = self._explore(scout_units[i], enemy_clusters, ally_clusters)

        return ndarray_to_moves(scout_moves)


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
def exploration_force(curr_pt, enemy_pts, ally_pts=None):
    if ally_pts is None:
        ally_force = 0
    else:
        ally_v = curr_pt - ally_pts
        ally_force = inverse_force(ally_v)
    enemy_v = curr_pt - enemy_pts
    enemy_force = inverse_force(enemy_v)
    return ally_force + SCOUT_ENEMY_SCALE * enemy_force

def border_repulsion(curr_pt, xmax, ymax):
    border_pts = np.array([[0, curr_pt[1]], [xmax, curr_pt[1]], [curr_pt[0], 0], [curr_pt[0], ymax]])
    border_v = curr_pt - border_pts
    return inverse_force(border_v)

def inverse_force(v):
    mag = np.sqrt((v ** 2).sum(axis=1, keepdims=True))
    force = (v / (mag+1e-7) / (mag+1e-7)).sum(axis=0)
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

def get_pressure_level(force: List[float]) -> int:
    p = np.linalg.norm(force)

    if p <= PRESSURE_LO_THRESHOLD:
        return PRESSURE_LO
    elif p > PRESSURE_LO_THRESHOLD and p < PRESSURE_HI_THRESHOLD:
        return PRESSURE_MID
    else:
        return PRESSURE_HI


# -----------------------------------------------------------------------------
#   Helper functions
# -----------------------------------------------------------------------------

def get_moves(unit_pos: List[Tuple[float, float]], target_loc: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Returns a list of 2-tuple (dist, angle) required to move a list of points
    from @unit_pos to @target_loc.
    """
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
    """Converts a list of shapely.geometry.Point into a list of 2-tuple of floats."""
    return list(map(shapely_pt_to_tuple, points))


def shapely_pt_to_tuple(point: Point) -> Tuple[float, float]:
    """Converts a shapely.geometry.Point into a 2-tuple of floats."""
    return ( float(point.x), float(point.y) )


def midsort(arr: List[float]) -> List[float]:
    """Sorts an array by repeatedly selecting the midpoints."""
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

        The map of the voronoi game is a 100 * 100 grid. From the top left going
        counter-clockwise are player p1 (index: 0), p2 (1), p3 (2), p4 (3).

        Below is a visualization of player 3's base angles.
        
            (base angle - pi/2) 
                ^
                | 
                |     
                -------> (base angle)
                    pi/2 * (1 - player_index)
              p2
    """
    base = (1 - player_idx) * math.pi / 2

    return base, base - math.pi / 2

def ndarray_to_moves(moves: List[List[float]]) -> List[Tuple[float, float]]:
    """Converts numpy adarray into list of 2-tuple of floats.
    
    Only 2-tuple of floats are accepted as valid actions by the simulator.
    """
    return list(map(tuple, moves))