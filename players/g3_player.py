from abc import ABC, abstractmethod
from functools import reduce
import logging
import math
from typing import Tuple, List, Dict
import time

import numpy as np
import ot
from shapely.geometry import Point


# -----------------------------------------------------------------------------
#   Player Parameters
# -----------------------------------------------------------------------------

LOG_LEVEL = logging.DEBUG

WALL_DENSITY = 0.1
WALL_RATIO = 0
PRESSURE_HI_THRESHOLD = 3
PRESSURE_LO_THRESHOLD = 1.5
PRESSURE_LO, PRESSURE_MID, PRESSURE_HI = range(3) 

SCOUT_HOMEBASE_SCALE = 10.0
SCOUT_BORDER_SCALE = 5.0
SCOUT_ENEMY_BASE_SCALE = 50.0

COOL_DOWN = 5
CB_DURATION = 5 # days dedicated to border consolidation in each cycle
CB_START = 35    # the day to start the first cycle of border consolidation


# -----------------------------------------------------------------------------
#   Custom Types
# -----------------------------------------------------------------------------

Tid = str
Uid = int
Upos = Tuple[float, float]
Move = Tuple[Uid, Upos]


# -----------------------------------------------------------------------------
#   Role Interface & Template 
# -----------------------------------------------------------------------------

class Player(ABC):
    map_states: np.ndarray
    homebase: np.ndarray
    enemy_bases: np.ndarray
    our_units: np.ndarray
    enemy_units: np.ndarray

    @abstractmethod
    def play(self):
        pass


class Role(ABC):
    @abstractmethod
    def select(self):
        pass

    @abstractmethod
    def move(self) -> List[Move]:
        pass


class RoleTemplate(Role):
    _logger: logging.Logger
    _player: Player
    _name:   Tid

    def __init__(self, logger: logging.Logger, player: Player, name: Tid):
        self._player = player
        self._name = name
        self._logger = logger

    def _debug(self, *args):
        self._logger.debug(
            f"[ {self._name} ] " + " ".join(str(a) for a in args))

    @property
    def name(self):
        return self._name

    @property
    def player(self):
        return self._player


# -----------------------------------------------------------------------------
#   Soldier Allocation, Deallocation, Coordination Framework
# -----------------------------------------------------------------------------

class ResourcePool:
    def __init__(self, player: Player):
        self.player = player
        self.unit_to_team_dict = {}
        self.team_to_unit_dict = {}
    
    # team_id = 'team''number' i.e. specialforce1

    def get_team_ids(self, team_id: Tid) -> List[Uid]:
        pass

    def get_team(self, unit_id: Uid) -> Tid:
        pass

    def is_dead(self, unit_id: Uid) -> bool:
        pass

    def get_team_casualties(self, team_id: Tid) -> List[Uid]:
        pass

    def get_free_units(self) -> List[Uid]:
        pass

    def claim_units(self, team_id: Tid, units: List[Uid]) -> List[Uid]:
        """Returns list of units unable to be claimed."""
    
    def release_units(self, team_id: Tid, units: List[Uid]) -> List[Uid]:
        """"Returns list of units who weren't successfully fired. They were not on the team in the first place"""
    
    def update_state(self):

        pass

    def get_positions(self, units: List[Uid]) -> List[List[float]]:
        pass


# -----------------------------------------------------------------------------
#   Enemy Pressure Estimator(s)
# -----------------------------------------------------------------------------

class DensityMap:

    def __init__(
        self,
        player_id: int,
        unit_pos: List[List[Tuple[float, float]]],
        grid_size = 10):

        max_dim = 99

        self.me = player_id
        self.grid_size = grid_size
        self.dmap_max_dim = math.ceil(max_dim / grid_size)

        self.soldier_partitions = self._partition_soldiers(unit_pos)
        self._dmap = self.__dmap()
        self._ndmap = self.__ndmap()

    def pt2grid(self, x: float, y: float) -> tuple[int, int]:
        """Given a location on map, return the grid it is in. The size
        of a grid could be access from self.grid_size.
        """

        row_id = math.floor(x / self.grid_size)
        col_id = math.floor(y / self.grid_size)

        return (row_id, col_id)

    def _partition_soldiers(self, unit_pos: List[List[Tuple[float, float]]]) -> Dict:
        """Partitions soldiers into n x n grids where n = self.grid_size.
        
        Returns a dictionary whose key is the grid identifier :: Tuple[int, int], and
        value is the list of soldiers :: Tuple[Tuple[float, float], int], where the
        first component is position, and the second indicates which player owns it.
        """

        partitions = dict()

        for player_id, army_locations in enumerate(unit_pos):
            for loc in army_locations:
                grid_key = self.pt2grid(loc[0], loc[1])
                if grid_key not in partitions:
                    partitions[grid_key] = []
                
                partitions[grid_key].append((loc, player_id))

        return partitions

    def __dmap(self) -> np.ndarray:
        """Computes the danger in each grid using soldier partitions.
        
        Let d(g) represents the danger of a grid, it is defined as
                    d(g) = n - m
        where n = # of enemeies in the grid
              m = # of allies in the grid
        
        A positive d(g) means we are outnumbered by enemies in grid g.
        The higher the value, the more outnumbered we are and dangerous.
        """
        danger_map = np.zeros((self.dmap_max_dim, self.dmap_max_dim))

        for x in range(self.dmap_max_dim):
            for y in range(self.dmap_max_dim):
                danger_map[x, y] = reduce(
                    lambda acc, el: acc - 1 if el[1] == self.me else acc + 1,
                    self.soldier_partitions.get((x, y), []),
                    0
                )
        
        return danger_map

    def __ndmap(self) -> np.ndarray:
        """Computes holisitc danger value of each grid.

        Let h(g) be the holisitc danger of a grid g, it is defined as
                h(g) = sum of [ alpha ** dist(g, g') * d(g') ]
        where d(g') is the danger value of grid g'
              dist(g, g') is the distance between the center of g and g'
              alpha is a distance scaling factor of 0.9
        """

        def holistic_danger(dmap, x, y):
            x_max = y_max = self.dmap_max_dim - 1
            alpha = 0.9
            val = 0

            for neighbor_x in {max(0, x-1), x, min(x_max, x+1)}:
                for neighbor_y in {max(0, y-1), y, min(y_max, y+1)}:
                    dist2neighbor = math.sqrt((x-neighbor_x)**2 + (y-neighbor_y)**2)
                    val += (alpha ** dist2neighbor) * dmap[neighbor_x][neighbor_y]
            
            return val

        hg_map = np.zeros((self.dmap_max_dim, self.dmap_max_dim))
        for x in range(self.dmap_max_dim):
            for y in range(self.dmap_max_dim):
                hg_map[x, y] = holistic_danger(self._dmap, x, y)

        return hg_map

    @property
    def dmap(self):
        return self._dmap

    @property
    def ndmap(self):
        return self._ndmap

    def pressure_level(self, pos: Tuple[float, float]) -> int:
        """Returns the pressure level at @pos based on holistic danger of the
        grid @pos is in and that of its neighboring grid.
        
        A *positive* holistic danger value h(g) means that it's likely that
        in grid we'll soon be outnumbered, indicating high pressure.

        Let g be the grid one of our soldier at @pos is in, and G' the set of
        8 immediately neighboring grids (or less if g is on border).

            h(g) >= 0                                 -> PRESSURE_HI
            h(g) < 0 and h(g') >= 0 for any g' in G   -> PRESSURE_MID
            otherwise                                 -> PRESSURE_LO
        """
        x, y = self.pt2grid(pos[0], pos[1])
        cell_dangerous = self.ndmap[x, y] >= 0
        neighbor_cell_dangerous = False
        
        x_max = y_max = self.dmap_max_dim - 1
        for neighbor_x in {max(0, x-1), x, min(x_max, x+1)}:
            for neighbor_y in {max(0, y-1), y, min(y_max, y+1)}:
                if not (neighbor_x == x and neighbor_y == y) and self._ndmap[neighbor_x, neighbor_y] >= 0:
                    neighbor_cell_dangerous = True
                    break

        if cell_dangerous:
            return PRESSURE_HI
        elif not cell_dangerous and neighbor_cell_dangerous:
            return PRESSURE_MID
        else:
            return PRESSURE_LO

    def suggest_move(self, ally_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Suggests a direction to move a soldier within a grid cell.

        If a soldier experiences PRESSURE_MID:
            a) more allies than enemies, soldier attacks intruders in the grid
            b) less allies than enemies, soldier retreats to be closer than ally
        
        We want a greater attraction force to enemies within a grid cell for (a),
        while we want a greater attraction foce to allyies for (b).

        Moreover, when there's no enemy in the grid, to avoid allies being squished
        together, we apply replly force if allies are too close. This helps space
        out allies in a grid.
        
        To achieve this, we need scale attraction force for enemies *inversely*
        with its enemy vs ally ratio, i.e.
          a) more enemies: want ally attract each other more to form groups, so we
             scale attration force of enemies less than that of allies.
          b) less enemies: want ally attracted to enemies to attack, so we scale
             attraction force of enemies larger than that of allies.

        Currently, an inverse ratio of ally2enemy number is used
        to scale attraction force of allies and enemies. TODO: a better metric for scale.
        """

        ally_pos = np.array(ally_pos)
        grid_id = self.pt2grid(ally_pos[0], ally_pos[1])
        troops = self.soldier_partitions[grid_id]
        ally2enemy_ratio = reduce(
            lambda acc, el: (acc[0] + 1, acc[1]) if el[1] == self.me else (acc[0], acc[1] + 1),
            troops,
            (0, 0)
        )

        enemy_attr_scale, ally_attr_scale = ally2enemy_ratio
        fvec = np.zeros((2,), dtype=float)

        if ally2enemy_ratio[1] == 0:
            # no enemy in the grid cell, make sure our allies are *spaced out*
            for other_ally, _ in troops:
                if (other_ally == ally_pos).all():
                    continue

                other_ally, me = np.array(other_ally), ally_pos
                dist2ally = np.linalg.norm(other_ally - me)

                if dist2ally < 1:
                    # ally within the same cell, REPELL!
                    fvec += 10 * repelling_force(ally_pos, other_ally)
                else:
                    # SPACE OUT A BIT!
                    fvec += repelling_force(ally_pos, other_ally)
        else:
            # has enemy(s) within the grid cell
            for other_soldier, pid in troops:
                if not (other_soldier == ally_pos).all():
                    attr_scale = ally_attr_scale if pid == self.me else enemy_attr_scale
                    fvec += attr_scale * attractive_force(ally_pos, other_soldier)

        angle = np.arctan2(fvec[1], fvec[0])
        return (1, angle)


# -----------------------------------------------------------------------------
#   Role Implementations
# -----------------------------------------------------------------------------

class DefaultSoldier(RoleTemplate):
    def __init__(self, logger: logging.Logger, player: Player, name: Tid, angles: List[float]):
        super().__init__(logger, player, name)

        self.counter = 0
        self.angles = angles
        self.unit_pos = []

    def _move_radially(self, pt: List[float], forward=True) -> Upos:
        """Moves @pt radially away from homebase, returns a tuple (distance, angle)
        to move the point.
        
        If @pt is at homebase, select an angle from self.midsorted_outer_wall_angles
        to move in.

        If @forward is True, move away from the homebase. Otherwise, towards the homebase.
        """
        homebase = self.player.homebase

        direction = 1 if forward else -1
        if (pt == homebase).all():
            angle = self.angles[self.counter % len(self.angles)]
            self.counter += 1
        else:
            towards_x, towards_y = np.array(pt) - np.array(homebase)
            angle = np.arctan2(towards_y, towards_x)
        
        return (direction, angle)

    def _push_radially(self, pt: List[float], plevel=False) -> Upos:
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

    def select(self):
        idx_taken = self.player.scout_team.unit_idx  # TODO: change later
        self.unit_pos = np.delete(self.player.our_units, idx_taken, axis=0)

    def move(self) -> List[Move]:
        dmap = self.player.d
        allies = self.unit_pos

        pressure_levels = [
            dmap.pressure_level(tuple(pos))
            for pos in allies
        ]
        soldier_moves = [
            self._push_radially(allies[i], plevel=plevel)
            if plevel != PRESSURE_MID else dmap.suggest_move(allies[i])
            for i, plevel in enumerate(pressure_levels)
        ]

        return soldier_moves


class Scouts(RoleTemplate):
    def __init__(self, logger: logging.Logger, player: Player, name: Tid, size: int):
        super().__init__(logger, player, name)

        self.target_size = size
        self.actual_size = 0

        self.unit_idx = []
        self.unit_pos = []

    def _get_clusters(self, ally_units):
        enemy_units = self.player.enemy_units

        # change to random selection to speed up
        num_enemies = enemy_units.shape[0]
        num_allies = ally_units.shape[0]

        enemy_clusters = enemy_units[np.random.choice(np.arange(num_enemies), min(num_enemies, 50), replace=False)]
        ally_clusters = ally_units[np.random.choice(np.arange(num_allies), min(num_allies, 15), replace=False)]

        return enemy_clusters, ally_clusters

    def _explore(self, scout_unit, enemy_clusters, ally_clusters):
        # player attributes
        map_states = self.player.map_states
        homebase = self.player.homebase
        enemy_bases = self.player.enemy_bases

        # push and pull factors
        explore_force = exploration_force(scout_unit, enemy_clusters, ally_pts=ally_clusters)
        homebase_force = inverse_force((scout_unit - homebase).reshape(1, 2))
        border_repelling_force = border_repulsion(scout_unit, xmax=map_states.shape[0], ymax=map_states.shape[1])
        enemy_base_attr = enemy_base_attraction(scout_unit, enemy_bases)
        
        # compute aggregate force
        force = (
            explore_force                               +
            border_repelling_force * SCOUT_BORDER_SCALE +
            homebase_force * SCOUT_HOMEBASE_SCALE       +
            enemy_base_attr * SCOUT_ENEMY_BASE_SCALE
        )

        return np.array([1, np.arctan2(force[1], force[0])])

    def select(self):
        """
        Dynamically select scouts based on distance from homebase.
        But we want them evenly spread out, so I changed it back to original until I figure out how to do it
        """

        self._debug("Scouts.select")
        self.actual_size = min(self.player.our_units.shape[0], self.target_size)
        self.unit_idx = np.arange(self.actual_size)
        self.unit_pos = self.player.our_units[self.unit_idx]

    def move(self) -> List[Move]:

        self._debug("Scouts.move")

        scout_units = self.unit_pos
        ally_units = np.delete(self.player.our_units, self.unit_idx, axis=0)
        enemy_units = self.player.enemy_units

        # compute ally and enemey distances
        ally_dist = ((scout_units.reshape(-1, 1, 2) - ally_units.reshape(1, -1, 2)) ** 2).sum(axis=2)
        enemy_dist = ((scout_units.reshape(-1, 1, 2) - enemy_units.reshape(1, -1, 2)) ** 2).sum(axis=2)
        
        min_ally_id = ally_dist.argmin(axis=1)
        min_enemy_dist = enemy_dist.min(axis=1)

        # compute moves for each scout
        scout_moves = np.zeros_like(scout_units, dtype=float)
        for i in range(self.actual_size):
            if ally_dist[i, min_ally_id[i]] >= min_enemy_dist[i] * 2:
                # retreat
                to_x, to_y = ally_units[min_ally_id[i]] - scout_units[i]
                scout_moves[i] = np.array([1, np.arctan2(to_y, to_x)])
            else:
                # explore
                enemy_clusters, ally_clusters = self._get_clusters(ally_units)
                scout_moves[i] = self._explore(scout_units[i], enemy_clusters, ally_clusters)

        return ndarray_to_moves(scout_moves)


class MacroArmy(RoleTemplate):

    def __init__(self, logger, player, name: Tid, resource: ResourcePool):
        super().__init__(logger, player, name)

        self.resource = resource
        self.unit_ids = None
        self.unit_pos = None
        self.targets = None
        self.MAX_UNITS = 500

    def select(self):
        # it would be good if get_free_units() returns an array and claim_units() takes input an array
        free_units = np.array(self.resource.get_free_units(), dtype=int) 
        self.unit_ids = np.random.choice(free_units, size=min(self.MAX_UNITS, free_units.shape[0]), replace=False)
        self.resource.claim_units(self.name, self.unit_ids.tolist())

    def move(self) -> List[Move]:
        if self.move == None:
            # Only calculate border and OT assignments once at creation
            self.unit_pos = np.array(self.resource.get_positions(self.unit_ids))
            border = self.resource.player.get_border()
            selected_border = border[np.random.choice(np.arange(border.shape[0]), size=min(border.shape[0], troops.shape[0]), replace=False)]
            self.targets = assign_by_ot(self.unit_pos, selected_border)
        else:
            # remove dead units without changing assignments for other units
            dead_units = []
            for i, unit_id in enumerate(self.unit_ids):
                if self.resource.is_dead(unit_id):
                    dead_units.append(i)
            self.unit_ids = np.delete(self.unit_ids, dead_units, axis=0)
            self.unit_pos = np.delete(self.unit_pos, dead_units, axis=0)
            self.targets = np.delete(self.targets, dead_units, axis=0)

        return list(zip(self.unit_id, get_moves(self.unit_pos, self.targets)))


class SpecialForce:
    def __init__(self, logger: logging.Logger, player_id, id, team_size: int, unit_idxs: List[int] = [], unit_pos: np.ndarray = np.array([])):
        self.player_id = player_id
        self.id = id
        self.team_size = team_size
        self.enemy = None
        self.logger = logger
        self.unit_pos_next_step = []
        if len(unit_idxs) == len(unit_pos):
            self.unit_idxs = unit_idxs[0:self.team_size]
            self.unit_pos = unit_pos[0:self.team_size]
        else:
            self.logger.info(f"SPECIAL FORCE {self.id}: length of provided unit idxs and unit position list arguments do not match defaulting to intiializing with no units")
            self.unit_idxs = []
            self.unit_pos = []

        # is team in the correct formation?
        self.in_formation = False
        self.formation = self.__create_formation()

        if len(unit_idxs) > self.team_size:
            self.logger.info(f"SPECIAL FORCE {self.id}: initialized special force team {self.id} with too many units")

    def add_unit(self, unit_idx: int):
        if len(self.unit_idxs) >= self.team_size:
            self.logger.info(f"SPECIAL FORCE {self.id}: cannot add unit {unit_idx} to special force team {self.id}. Too many units")
            return False
        
        self.unit_idxs.append(unit_idx)
        print(self.unit_idxs)
        return True

    def get_unit_idxs(self):
        return self.unit_idxs
    
    def is_team_full(self):
        return len(self.unit_idxs) >= self.team_size

    def set_target_enemy(self, enemy: List[float]):
        self.enemy = enemy
    
    def __create_formation(self):
        precomp_formation = [[0, 0]]
        concentric_circle_points = [[1, 0]] # radius, points

        for i in range(self.team_size - 1):
            circle_0_radius = concentric_circle_points[0][0]
            circle_0_point_count = concentric_circle_points[0][1]
            min_density_circle_idx = 0
            max_space_between_points = 0

            for circle_idx in range(len(concentric_circle_points)):
                circle_radius, point_count = concentric_circle_points[circle_idx]
                
                circumference = 2 * np.pi * circle_radius
                if point_count == 0 or point_count == 1:
                    space_between_points = circumference
                elif point_count == 2:
                    space_between_points = 2 * circle_radius
                else:
                    angle = (2 * np.pi) / point_count
                    space_between_points = 2 * circle_radius * np.sin(angle / 2)
                
                if space_between_points > max_space_between_points:
                    max_space_between_points = space_between_points
                    min_density_circle_idx = circle_idx

            if 1 / len(concentric_circle_points) < max_space_between_points:
                # add point to min density circle
                concentric_circle_points[min_density_circle_idx][1] += 1
            else:
                # readjust radius and start new inner radius with 3 points
                inner_radius = 1 / (len(concentric_circle_points) + 1)
                concentric_circle_points.insert(0, [inner_radius, 1])

                if len(concentric_circle_points) == 2:
                    concentric_circle_points[1][1] -= 2
                    concentric_circle_points[0][1] += 2
                
                else:
                    for circle_idx in range(1, min(len(concentric_circle_points), 6)):
                        concentric_circle_points[0][1] += 1
                        concentric_circle_points[circle_idx][1] -= 1
                

                for circle_idx in range(1, len(concentric_circle_points)):
                    concentric_circle_points[circle_idx][0] = concentric_circle_points[circle_idx - 1][0] + inner_radius

                concentric_circle_points[-1][0] = 1
        
        # calculate points for each circle
        for circle_radius, point_count in concentric_circle_points:
            for theta in np.linspace(0, 2 * np.pi, point_count + 1)[0:point_count]:
                precomp_formation.append([circle_radius * np.cos(theta), circle_radius * np.sin(theta)])
        
        return np.array(precomp_formation)

    def __compute_formation_positions_around_centroid(self, centroid: np.ndarray):
        positions = np.add(centroid, self.formation)
        return np.clip(positions, 0, 100)

    def check_in_formation(self):
        # check if all units in formation
        if len(self.unit_pos) == 0:
            return False
        # np.array_equal(self.unit_pos, self.__compute_formation_positions_around_centroid(centroid = self.unit_pos[0]))
        
        return np.all(np.absolute(np.subtract(self.unit_pos, self.__compute_formation_positions_around_centroid(centroid = self.unit_pos[0])[0: len(self.unit_pos)])) < [0.01, 0.01])
    
    def __congregate(self):
        # we define target_team_centroid as the centroid of the meeting location of all soldiers
        if not self.is_team_full():
            # if team not done being built yet, congregate near home base at 1, 1
            if self.player_id == 0:
                target_team_centroid = np.array([1, 1])
            elif self.player_id == 1:
                target_team_centroid = np.array([1, 99])
            elif self.player_id == 2:
                target_team_centroid = np.array([99, 99])
            else:
                target_team_centroid = np.array([99, 1])
        else:
            target_team_centroid = np.clip(np.average(np.subtract(self.unit_pos, self.formation[0:len(self.unit_pos)]), axis = 0), 1, 99)
        
        self.unit_pos_next_step = self.__compute_formation_positions_around_centroid(centroid = target_team_centroid)[0: len(self.unit_idxs)]

    def __attack_target_enemy(self):
        cur_centroid = np.array(self.unit_pos[0])
        unit_vec_towards_enemy = (np.subtract(self.enemy, cur_centroid)) / np.linalg.norm(self.enemy / cur_centroid)
        self.unit_pos_next_step = self.__compute_formation_positions_around_centroid(centroid = cur_centroid + unit_vec_towards_enemy)[0: len(self.unit_idxs)]

    def move(self):
        if self.in_formation:
            print("attacking")
            self.__attack_target_enemy()
        else:
            print("congregateing")
            self.__congregate()
        
        if len(self.unit_pos) == 0:
            return None
        # returns [ [unit_idx, (distance, angle)], ... ]
        return zip(self.unit_idxs, get_moves(self.unit_pos, self.unit_pos_next_step[0:len(self.unit_pos)]))

    def update_state(self, unit_pos:  np.ndarray):
        assert(len(unit_pos) == len(self.unit_idxs))
        self.unit_pos = unit_pos
        self.in_formation = self.check_in_formation()


# -----------------------------------------------------------------------------
#   G3 Player
# -----------------------------------------------------------------------------

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

        self.rng = rng
        self.logger = logger
        self.logger.setLevel(LOG_LEVEL)

        self.us = player_idx
        self.homebase = np.array(spawn_point)
        self.enemy_bases = np.delete(np.array([[0.5, 0.5], [0.5, 99.5], [99.5, 99.5], [99.5, 0.5]]), self.us, axis=0)
        self.day_n = 0

        self.our_units = None
        self.enemy_units = None
        self.enemy_offsets = None
        self.map_states = None
        self.border = None

        self.target_loc = []

        self.initial_radius = 35
        self.num_scouts = 3

        base_angles = get_base_angles(player_idx)
        outer_wall_angles = np.linspace(start=base_angles[0], stop=base_angles[1], num=int(self.initial_radius * 2 / 1.4))
        self.counter = 0
        self.midsorted_outer_wall_angles = midsort(outer_wall_angles)

        self.cb_scheduled = np.array([CB_START, CB_START + CB_DURATION])

        # compute special forces metadata
        self.total_lifetime_units = total_days // spawn_days
        self.units_by_day_50 = 50 // spawn_days
        self.sf_units = self.total_lifetime_units / 5
        self.sf_units_per_team = max(self.sf_units // 5, 10)
        self.sf_units_per_team = min(self.sf_units_per_team, self.sf_units)

        # SAMPLE SQUAD
        '''
        self.special_forces = [SpecialForce(self.logger, self.us, id=i, team_size=13) for i in range(1)]
        self.special_forces[0].set_target_enemy([20, 25])'''
        # round down sf_units to nearest multiple of sf_units per team
        self.sf_units -= self.sf_units % self.sf_units_per_team
        self.sf_teams = self.sf_units // self.sf_units_per_team

        # Temporary - Scout, DefaultSoldier
        self.scout_team = Scouts(self.logger, self, 'scouts1', 3)
        self.default_soldiers = DefaultSoldier(self.logger, self, 'default1', self.midsorted_outer_wall_angles)

    def debug(self, *args):
        self.logger.info(" ".join(str(a) for a in args))

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

        float_unit_pos = [shapely_pts_to_tuples(pts) for pts in unit_pos]
        self.day_n += 1
        self.map_states = np.array(map_states) - 1
        self.enemy_offsets = np.array([len(unit_pos[i]) for i in range(4) if i != self.us])
        self.enemy_units = np.concatenate([float_unit_pos[i] for i in range(4) if i != self.us])
        self.our_units = np.array(float_unit_pos[self.us])
        self.d = DensityMap(self.us, float_unit_pos)
        self.our_unit_ids = np.array(unit_id[self.us], dtype=int)


        self.debug()
        self.debug(f'unit_ids: {unit_id[self.us]}')
        self.debug(f'len(unit_pos): {len(unit_pos[self.us])}, len(unit_ids): {len(unit_id[self.us])}')
        self.debug(f'density map: {self.d.dmap.T}')
        self.debug(f'average neighbor density: {self.d.ndmap.T}')


        if self.day_n < self.initial_radius:
            self.debug(f'day {self.day_n}: form initial wall')

            while len(unit_id[self.us]) > len(self.target_loc):
                # add new target_locations
                self.target_loc.append(order2coord(
                        self.homebase,
                        [self.initial_radius, self.midsorted_outer_wall_angles[len(unit_id[self.us]) - 1]]
                ))
        
            return get_moves(shapely_pts_to_tuples(unit_pos[self.us]), self.target_loc)
        elif self.day_n >= self.cb_scheduled[0] and self.day_n < self.cb_scheduled[1]:
            self.debug(f'day {self.day_n}: consoldiate border')

            if self.day_n == self.cb_scheduled[0] or (CB_DURATION >= 10 and self.day_n == self.cb_scheduled[0] + CB_DURATION // 2):
                self.border = self.get_border()

            if self.day_n == self.cb_scheduled[1] - 1:
                self.cb_scheduled += (COOL_DOWN + CB_DURATION)
            
            # allocation phase
            self.scout_team.select()
            scout_ids = self.scout_team.unit_idx  # TODO: change later

            # mobilization phase
            defense_moves = self.send_to_border(scout_ids, self.border)
            offense_moves = self.scout_team.move()
        else:
            # MID_GAME: adjust formation based on opponents' positions
            self.debug(f'day {self.day_n}: cool down')

            # allocation phase
            self.scout_team.select()
            self.default_soldiers.select()
            scout_ids = self.scout_team.unit_idx  # TODO: change later

            # mobilization phase
            start = time.time()
            defense_moves = self.default_soldiers.move()
            self.debug(f'Defense: {time.time()-start}s')
            
            start = time.time()
            offense_moves = self.scout_team.move()
            self.debug(f'Offense: {time.time()-start}s')

        # insert scout moves into all moves
        all_moves = defense_moves
        for i, scout in enumerate(scout_ids.tolist()):
            all_moves = all_moves[:scout] + [offense_moves[i]] + all_moves[scout:]
        return all_moves

    def send_to_border(self, scout_ids, border) -> List[Tuple[float, float]]:
        """Sends soldiers to consolidate our border."""
        troops = np.delete(self.our_units, scout_ids, axis=0)
        selected_border = border[np.random.choice(np.arange(border.shape[0]), size=min(border.shape[0], troops.shape[0]), replace=False)]
        targets = assign_by_ot(troops, selected_border)
        return get_moves(troops, targets)

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

        # remove pts on edge of map (i.e., not on frontline)
        frontline = []
        for pt in border:
            if not self._on_edge(pt):
                frontline.append(pt)
        return np.array(frontline)

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
                if neighbor != curr_pt and (self._on_border(neighbor) or self._on_edge(neighbor)):
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

    def _on_edge(self, pt):
        """Check if given point is on the edge of map"""
        if self.map_states[pt] != self.us: 
            return False
        xmax, ymax = self.map_states.shape
        return (pt[0] in [0, xmax-1] or pt[1] in [0, ymax-1])


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
    return ally_force + enemy_force

def border_repulsion(curr_pt, xmax, ymax):
    border_pts = np.array([[0, curr_pt[1]], [xmax, curr_pt[1]], [curr_pt[0], 0], [curr_pt[0], ymax]])
    border_v = curr_pt - border_pts
    return inverse_force(border_v)

def enemy_base_attraction(curr_pt, enemy_bases):
    """Squared inverse attraction to enemy homebases so that scouts attack enemey base only in proximity"""
    nearest_base = enemy_bases[((curr_pt - enemy_bases) ** 2).sum(axis=1).argmax()]
    v = nearest_base - curr_pt
    return inverse_force_cubic(v)

def inverse_force(v):
    mag = np.sqrt((v ** 2).sum(axis=1, keepdims=True)) + 1e-7
    force = (v / mag / mag).sum(axis=0)
    return force

def inverse_force_cubic(v):
    mag = np.sqrt((v ** 2).sum()) + 1e-7
    force = v / mag / (mag ** 3)
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

def attractive_force(p1: Tuple[float, float], p2: Tuple[float, float]) -> List[float]:
    return reactive_force(repelling_force(p1, p2))

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

def assign_by_ot(unit_pos, target_loc):
    """
    Assign troops to locations based on optimal transport.
    unit_pos - shape (N, 2)
    target_loc - shape (N, 2)
    Returns reordered target_loc optimally mapped to each unit - shape (N, 2)
    """
    a, b = np.ones((unit_pos.shape[0],)) / unit_pos.shape[0] , np.ones((target_loc.shape[0],)) / target_loc.shape[0]  # uniform weights on points
    M = ot.dist(unit_pos, target_loc, metric='sqeuclidean') # cost matrix
    assignment = ot.emd(a, b, M).argmax(axis=1) # OT linear program solver
    return target_loc[assignment]

def order2coord(homebase: np.ndarray, order: Tuple[float, float]) -> Upos:
    """Converts an order, tuple of (dist2homebase, angle), into a coordinate."""
    dist, angle = order
    x = homebase[0] + dist * math.cos(angle)
    y = homebase[1] + dist * math.sin(angle)

    return (x, y)

def get_moves(unit_pos, target_loc) -> List[Tuple[float, float]]:
    """Returns a list of 2-tuple (dist, angle) required to move a list of points
    from @unit_pos to @target_loc.
    """
    if type(unit_pos) == list:
        assert len(unit_pos) == len(target_loc), "get_moves: unit_pos and target_loc array length not the same"
        np_unit_pos = np.array(unit_pos, dtype=float)
        np_target_loc = np.array(target_loc, dtype=float)
    else:
        assert unit_pos.shape[0] == target_loc.shape[0], "get_moves: unit_pos and target_loc array length not the same"
        np_unit_pos = unit_pos
        np_target_loc = target_loc

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