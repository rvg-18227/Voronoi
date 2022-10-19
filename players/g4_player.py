import logging
import multiprocessing
import pdb
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from glob import glob
from itertools import repeat
from multiprocessing import parent_process
from os import makedirs, remove
from random import randint, uniform
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import colors
from matplotlib.collections import LineCollection
from numpy.typing import NDArray
from scipy.spatial import KDTree
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import nearest_points, voronoi_diagram
from sklearn.cluster import DBSCAN
from typing_extensions import TypeAlias

from constants import dispute_color, player_color, tile_color

# ==================
# Game State Classes
# ==================

attack_roll = 0


class GameParameters:
    """Represents constant game parameters that don't change."""

    total_days: int
    spawn_days: int
    player_idx: int
    spawn_point: tuple[float, float]
    min_dim: int
    max_dim: int
    home_base: tuple[float, float]


def get_nearest_unit(
    args: tuple[tuple[int, int], int, KDTree, dict[int, str]]
) -> tuple[tuple[int, int], int, str]:
    pos, player, kdtree, idx_to_id = args
    tile_x, tile_y = pos
    # Find closest unit
    # KDTree returns the index of the nearest point from the input list of points
    _, closest_unit_idx = kdtree.query((tile_x + 0.5, tile_y + 0.5))
    # Convert that index to the unit id
    return pos, player, idx_to_id[closest_unit_idx]


# Pool initialization must be below the declaration for get_nearest_unit
# Pool initialization is global to re-use the same process pool
THREADED = True
if THREADED:
    pool = multiprocessing.Pool(multiprocessing.cpu_count())


class StateUpdate:
    """Represents all of the data that changes between turns."""

    params: GameParameters
    unit_id: list[list[str]]
    unit_pos: list[list[NDArray[np.float32]]]
    map_states: list[list[int]]
    turn: int
    player_unit_id_to_pos: dict[tuple[int, str], NDArray[np.float32]]
    player_unit_idx_to_id: dict[int, dict[int, str]]
    cached_ownership: Optional[
        tuple[
            dict[int, dict[str, list[tuple[int, int]]]],
            dict[tuple[int, int], tuple[int, str]],
        ]
    ]
    cached_clusters: Optional[dict[int, dict[int, list[str]]]]

    def __init__(
        self,
        params: GameParameters,
        turn: int,
        unit_id: list[list[str]],
        unit_pos: list[list[NDArray[np.float32]]],
        map_states: list[list[int]],
    ):
        self.params = params
        self.turn = turn
        self.unit_id = unit_id
        self.unit_pos = unit_pos
        self.map_states = map_states

        self.player_unit_id_to_pos = {
            (player, uid): pos
            for player in range(4)
            for uid, pos in zip(unit_id[player], unit_pos[player])
        }
        self.player_unit_idx_to_id = {
            player: {idx: uid for idx, uid in enumerate(self.unit_id[player])}
            for player in range(4)
        }
        self.cached_ownership = None
        self.cached_clusters = None

    # =============================
    # Role Update Utility Functions
    # =============================

    def enemies(self):
        return set(range(4)) - set([self.params.player_idx])

    def own_units(self):
        """Returns dictionary of `unit_id -> unit_pos` for this player's units."""

        return {
            unit_id: unit_pos
            for unit_id, unit_pos in zip(
                self.unit_id[self.params.player_idx],
                [pos for pos in self.unit_pos[self.params.player_idx]],
            )
        }

    def enemy_units(self):
        """Returns dictionary of `enemy player -> [(unit_id, unit_pos)]`."""

        return {
            enemy_id: list(zip(self.unit_id[enemy_id], self.unit_pos[enemy_id]))
            for enemy_id in self.enemies()
        }

    def all_enemy_units(self):
        """Returns all enemy units in a list `[(unit_id, unit_pos)]`."""
        return [
            unit for enemy_units in self.enemy_units().values() for unit in enemy_units
        ]

    def unit_id_to_pos(self, player: int, unit_id: str) -> NDArray[np.float32]:
        return self.player_unit_id_to_pos[(player, unit_id)]

    def unit_idx_to_id(self, player: int, idx: int) -> str:
        return self.player_unit_idx_to_id[player][idx]

    def unit_ownership(self):
        """
        Returns tile/unit ownership information `(unit_to_owned, tile_to_unit)`.

        `unit_to_owned`: dict of dicts `player_idx -> unit_id -> [(tile_x, tile_y)]`
        `tile_to_unit`: dict of `(tile_x, tile_y) -> (player_idx, unit_id)`
        """
        if self.cached_ownership is not None:
            return self.cached_ownership

        # player -> unit -> [(x, y)]
        unit_to_owned: dict[int, dict[str, list[tuple[int, int]]]] = {
            0: {},
            1: {},
            2: {},
            3: {},
        }
        # (x, y) -> (player, unit)
        tile_to_unit: dict[tuple[int, int], tuple[int, str]] = {}

        player_unit_idx_to_id = {
            player: {idx: uid for idx, uid in enumerate(self.unit_id[player])}
            for player in range(4)
        }
        player_kdtrees = {player: KDTree(self.unit_pos[player]) for player in range(4)}

        work: list[tuple[tuple[int, int], int, KDTree, dict[int, str]]] = []
        for tile_x in range(self.params.max_dim):
            for tile_y in range(self.params.max_dim):
                tile_state = self.map_states[tile_x][tile_y]
                # Disputed tiles aren't owned by any unit
                if tile_state == -1:
                    continue

                # 1-4 -> 0-3
                owning_player = tile_state - 1
                work.append(
                    (
                        (tile_x, tile_y),
                        owning_player,
                        player_kdtrees[owning_player],
                        player_unit_idx_to_id[owning_player],
                    )
                )

        if THREADED:
            results = pool.map(get_nearest_unit, work)
        else:
            results = map(get_nearest_unit, work)

        for result in results:
            pos, owning_player, closest_uid = result
            if not closest_uid in unit_to_owned[owning_player]:
                unit_to_owned[owning_player][closest_uid] = []
            unit_to_owned[owning_player][closest_uid].append(pos)

            tile_to_unit[pos] = (owning_player, closest_uid)

        self.cached_ownership = (unit_to_owned, tile_to_unit)
        return self.cached_ownership

    def enemy_clusters(self):
        if self.cached_clusters is not None:
            return self.cached_clusters

        enemies = self.enemies()
        enemy_units = self.enemy_units()
        player_unit_idx_to_id = {
            enemy: {idx: uid for idx, uid in enumerate(self.unit_id[enemy])}
            for enemy in enemies
        }
        dbscans = {
            enemy: DBSCAN(eps=3, min_samples=2).fit(
                [unit_pos for _, unit_pos in enemy_units[enemy]]
            )
            for enemy in enemies
        }
        clusters: dict[int, dict[int, list[str]]] = {enemy: {} for enemy in enemies}
        for enemy in enemies:
            for unit_idx, label in enumerate(dbscans[enemy].labels_):
                if label == -1:
                    continue
                if not label in clusters[enemy]:
                    clusters[enemy][label] = []
                clusters[enemy][label].append(player_unit_idx_to_id[enemy][unit_idx])
        self.cached_clusters = clusters
        return clusters


# =======================
# Force Utility Functions
# =======================
EPSILON = 0.0000001


def point_to_floats(p: Point):
    return np.array([float(p.x), float(p.y)])


def force_vec(p1, p2):
    """Vector direction and magnitude pointing from `p2` to `p1`"""
    v = p1 - p2
    mag = np.linalg.norm(v)
    unit = v / (mag + EPSILON)
    return unit, mag


def to_polar(p):
    x, y = p
    return np.sqrt(x**2 + y**2), np.arctan2(y, x)


def normalize(v):
    return v / (np.linalg.norm(v) + EPSILON)


def repelling_force(p1, p2) -> NDArray[np.float32]:
    dir, mag = force_vec(p1, p2)
    # Inverse magnitude: closer things apply greater force
    return dir * 1 / (mag + EPSILON)


EASING_EXP = 5


def ease_in(x):
    if x > 1:
        return 1
    elif x < 0:
        return 0
    else:
        return x**EASING_EXP


def ease_out(x):
    if x > 1:
        return 0
    elif x < 0:
        return 1
    else:
        return 1 - ((1 - x) ** EASING_EXP)


# =================
# Unit Role Classes
# =================


class RoleType(Enum):
    DEFENDER = 1
    ATTACKER = 2
    SCOUT = 3
    INTERCEPTOR = 4


class Role(ABC):
    # Assigned when Role is inserted into RoleGroups
    id: str
    _logger: logging.Logger
    _params: GameParameters
    _allocated_units: list[str]

    def __init__(self, logger, params):
        self._logger = logger
        self._params = params

        self.id = ""
        self.__allocated_units = []

    def debug(self, *args):
        self._logger.info(" ".join(str(a) for a in args))

    # ===========
    # Role Common
    # ===========

    def allocate_unit(self, unit_id: str):
        self.__allocated_units.append(unit_id)

    def deallocate_unit(self, unit_id: str):
        self.__allocated_units = [
            unit for unit in self.__allocated_units if unit != unit_id
        ]

    @property
    def units(self):
        return self.__allocated_units.copy()

    @property
    def params(self):
        return self._params

    def turn_moves(self, update: StateUpdate):
        # TODO: This should be a play phase before doing anything else
        alive_units = set(update.own_units().keys())
        allocated_set = set(self.__allocated_units)
        dead_units = allocated_set - alive_units
        self.__allocated_units = list(allocated_set - dead_units)

        return self._turn_moves(update, dead_units)

    # ===================
    # Role Specialization
    # ===================

    @abstractmethod
    def _turn_moves(
        self, update: StateUpdate, dead_units: set[str]
    ) -> dict[str, NDArray[np.float32]]:
        """Returns the moves this turn for the units allocated to this role."""
        pass

    @abstractmethod
    def deallocation_candidate(
        self, update: StateUpdate, target_point: NDArray[np.float32]
    ) -> str:
        """Returns a suitable allocated unit to be de-allocated and used for other roles."""
        pass


class LatticeDefender(Role):
    __spawn_jitter: dict[str, NDArray[np.float32]]
    __radius: float
    counter: int

    def __init__(self, logger, params, radius):
        super().__init__(logger, params)
        self.__radius = radius
        self.__spawn_jitter = {}
        self.counter = 0

    def inside_radius(self, unit_pos: NDArray[np.float32]):
        dist = np.linalg.norm(unit_pos - self.params.home_base)
        if dist > self.__radius:
            in_vec, _ = force_vec(self.params.home_base, unit_pos)
            return in_vec
        return np.array([0, 0])

    def get_jitter(self, uid: str):
        # Add decaying random direction bias
        if not uid in self.__spawn_jitter:
            self.__spawn_jitter[uid] = np.array((uniform(0.1, 1), uniform(0, 0.9)))

        jitter_force = self.__spawn_jitter[uid]
        self.__spawn_jitter[uid] *= 0.5

        return jitter_force

    def _turn_moves(self, update, dead_units):
        self.counter += 1
        try:
            envelope = box(0, 0, self.params.max_dim, self.params.max_dim)
            points = MultiPoint(
                [Point(pos) for pos in update.unit_pos[self.params.player_idx]]
            )
            voronoi_polys = list(voronoi_diagram(points, envelope=envelope).geoms)

            # Credit: G1
            fixed_voronoi_polys = []
            for region in voronoi_polys:
                region_bounded = region.intersection(envelope)
                if region_bounded.area > 0:
                    fixed_voronoi_polys.append(region_bounded)

            # Visualize Voronoi regions
            # plt.clf()
            # for poly in fixed_voronoi_polys:
            #     plt.fill(
            #         *list(zip(*poly.exterior.coords)),
            #         facecolor="#ffffcc",
            #         edgecolor="black",
            #         linewidth=1,
            #     )
            # plt.savefig(f"debug/{self.counter}.png")

            defender_positions = {
                id: pos
                for id, pos in zip(
                    update.unit_id[self.params.player_idx],
                    update.unit_pos[self.params.player_idx],
                )
                if id in self.units
            }

            moves = {}
            found = 0
            for uid, pos in defender_positions.items():
                jitter_force = self.get_jitter(uid)

                unit_point = Point(pos[0], pos[1])
                target = pos
                for poly in fixed_voronoi_polys:
                    if poly.contains(unit_point):
                        found += 1
                        target = np.array([poly.centroid.x, poly.centroid.y])
                        continue
                moves[uid] = to_polar(
                    normalize(
                        jitter_force
                        + normalize(target - pos)
                        + (100 * self.inside_radius(pos))
                    )
                )
            return moves
        except Exception as e:
            self.debug("Exception when processing LatticeDefender moves", e)
            return {
                uid: to_polar(normalize(self.get_jitter(uid))) for uid in self.units
            }

    def deallocation_candidate(self, update, target_point):
        # TODO: don't need this if dead unit update is its own phase
        alive_units = set(self.units).intersection(set(update.own_units().keys()))

        kdtree = KDTree(
            [update.unit_id_to_pos(self.params.player_idx, uid) for uid in alive_units]
        )
        _, unit_idx = kdtree.query(target_point)
        return update.unit_idx_to_id(self.params.player_idx, unit_idx)


class Interceptor(Role):
    __target_player: int
    __targets: set[str]

    def __init__(self, logger, params, target_player: int, targets: list[str]):
        super().__init__(logger, params)

        self.__target_player = target_player
        self.__targets = set(targets)

        self.noise = dict()

    @property
    def targets(self):
        return list(zip(repeat(self.__target_player), self.__targets))

    def get_centroid(self, units):
        """
        Find centroid on a cluster of points
        """
        return units.mean(axis=0)

    def find_closest_point(self, line, point):
        """
        Find closest point on line segment given a point
        """
        return nearest_points(line, point)[0]

    def _turn_moves(self, update, dead_units):
        target_enemy_units = set(update.unit_id[self.__target_player])
        # Prune dead targets
        self.__targets = self.__targets.intersection(target_enemy_units)
        if len(self.__targets) == 0:
            return {unit: (0.0, 0.0) for unit in self.units}

        target_positions = [
            Point(update.unit_id_to_pos(self.__target_player, target))
            for target in self.__targets
        ]
        nearest_target_position, _ = nearest_points(
            MultiPoint(target_positions), Point(self.params.home_base)
        )
        nearest_target_position = point_to_floats(nearest_target_position)

        ATTACK_INFLUENCE = 100
        AVOID_INFLUENCE = 300

        moves = {}
        own_units = update.own_units()

        # get where to initiate the attack
        units_array = np.stack(
            [pos for uid, pos in own_units.items() if uid in self.units]
        )
        start_point = self.get_centroid(units_array)

        # get attack target and get formation
        towards_target, _ = force_vec(start_point, nearest_target_position)
        avoid = nearest_target_position
        target = nearest_target_position - towards_target * 5

        formation = LineString([start_point, target])

        # calcualte force
        for unit_id in self.units:
            unit_pos = own_units[unit_id]
            closest_pt_on_formation = self.find_closest_point(
                formation, Point(unit_pos)
            )

            attack_force = self.attack_point(unit_pos, target, closest_pt_on_formation)

            attack_repulsion_force = repelling_force(unit_pos, avoid)

            if unit_id not in self.noise:
                noise_force = self.noise_force(attack_force)
                self.noise[unit_id] = noise_force

            # pdb.set_trace()
            dist_to_avoid = np.linalg.norm(avoid - unit_pos)
            noise_influence = 1 / dist_to_avoid * 10

            total_force = normalize(
                attack_repulsion_force * AVOID_INFLUENCE
                + ATTACK_INFLUENCE * attack_force
                + noise_influence * self.noise[unit_id]
            )

            moves[unit_id] = to_polar(total_force)
        return moves

    def noise_force(self, attack_force):
        # generate noise force roughly in the same direction as attack force
        # pdb.set_trace()
        r = randint(0, 100)
        vec_perpendicular = attack_force.copy()
        if r > 50:
            vec_perpendicular[0] *= -1
        else:
            vec_perpendicular[1] *= -1
        return vec_perpendicular

    def attack_point(self, unit, target, closest_point):
        """Given a unit, attack the target point following the foramtion.
        Args:
            units: attack unit
            target: attack target
            cloest_points: cloest point a unit is from its formation
        Return:
            attack vector following the formation
        """
        unit_vec_closest, mag_closest = force_vec(unit, closest_point.coords)
        unit_vec_target, mag_target = force_vec(unit, target)
        # Calculate weight for cloest point and target point
        total_mag = mag_target + mag_closest + EPSILON
        weight_target = mag_target / total_mag
        weight_closest = mag_closest / total_mag
        # Calculate move vec for each units
        attack_vec = unit_vec_closest * weight_closest + unit_vec_target * weight_target
        attack_vec = attack_vec
        attack_vec *= -1
        return attack_vec[0]

    def deallocation_candidate(self, update, target_point):
        pass


class RadialDefender(Role):
    __radius: float
    # TODO: Currently assuming that newly added units are coming from spawn
    # But they might be re-allocations from other roles
    __spawn_jitter: dict[str, NDArray[np.float32]]

    def __init__(self, logger, params, radius):
        super().__init__(logger, params)
        self.__radius = radius
        self.__spawn_jitter = {}

    @property
    def radius(self):
        return self.__radius

    def towards_radius(self, unit_pos: NDArray[np.float32]):
        dist = np.linalg.norm(unit_pos - self.params.home_base)
        if dist > self.__radius:
            in_vec, _ = force_vec(self.params.home_base, unit_pos)
            dr = dist - self.__radius
            # Easing between radius and 2 * radius, max out beyond 2 * radius
            return in_vec * ease_out(dr / (self.__radius * 2))
        else:
            out_vec, _ = force_vec(unit_pos, self.params.home_base)
            dr = self.__radius - dist
            # Easing between 0 and radsiu, maxing out at 0
            return out_vec * ease_in((self.__radius - dr) / self.__radius)

    def _turn_moves(self, update, dead_units):
        RADIAL_INFLUENCE = 5
        ALLY_INFLUENCE = 0.5

        moves = {}
        own_units = update.own_units()

        for unit_id in self.units:
            unit_pos = own_units[unit_id]

            # Add decaying random direction bias
            if not unit_id in self.__spawn_jitter:
                self.__spawn_jitter[unit_id] = np.array(
                    (uniform(0.1, 1), uniform(0, 0.9))
                )

            jitter_force = self.__spawn_jitter[unit_id]
            self.__spawn_jitter[unit_id] *= 0.5

            ally_forces = [
                repelling_force(unit_pos, ally_pos)
                for ally_id, ally_pos in own_units.items()
                if ally_id != unit_id and ally_id in self.units
            ]
            ally_force = np.add.reduce(ally_forces)

            # disable ally repulsion when traveling to designated radius
            dist_from_home = np.linalg.norm(unit_pos - self.params.home_base)
            dist_from_radius = np.abs(dist_from_home - self.__radius)
            if np.abs(dist_from_radius) > 2:
                ally_force *= 0

            radius_maintenance = self.towards_radius(unit_pos)

            total_force = normalize(
                ally_force * ALLY_INFLUENCE
                + radius_maintenance * RADIAL_INFLUENCE
                + jitter_force
            )

            moves[unit_id] = to_polar(total_force)

        return moves

    def deallocation_candidate(self, update, target_point):
        pass


class FirstScout(Role):
    def __init__(self, logger, params, scout_id):
        super().__init__(logger, params)

    def _turn_moves(self, update, dead_units):
        own_units = update.own_units()
        moves = {}
        for unit_id in self.units:
            unit_pos = own_units[unit_id]
            home_force = repelling_force(unit_pos, self.params.home_base)
            total_force = normalize((home_force * 10))
            # self._logger.debug("force", total_force)
            moves[unit_id] = to_polar(total_force)
            # self._debug(moves)

        return moves

    def deallocation_candidate(self, update, target_point):
        pass


class GreedyScout(Role):
    def __init__(self, logger, params):
        super().__init__(logger, params)
        self.owned = {}
        self.first_turn = True

        # unit_id -> scout_id
        self.temp_id = {}
        self.id_counter = 0

    def closest_enemy_dist(self, update, uid, own_units):
        enemy_units = update.all_enemy_units()
        p = own_units[uid]
        closest_dist = 500
        for _, enemy_pos in enemy_units:
            closest_dist = min(closest_dist, np.linalg.norm(p - enemy_pos))
        return closest_dist

    def _turn_moves(self, update, dead_units):
        for uid in self.units:
            if not uid in self.temp_id:
                self.temp_id[uid] = self.id_counter
                self.id_counter += 1

        HOME_INFLUENCE = 30
        own_units = update.own_units()
        moves = {}
        unit_to_owned, _ = update.unit_ownership()
        for unit_id in self.units:
            # self._debug("text")
            # self._debug(f"uid {unit_id}")
            # self._debug(f"owns {self.ownership}")
            # self._debug(f"owns 2 {self.ownership[self.params.player_idx]}")
            owns = 0
            if unit_id in unit_to_owned[self.params.player_idx]:
                owns = len(unit_to_owned[self.params.player_idx][unit_id])
            if not unit_id in self.owned:
                self.owned[unit_id] = owns
            else:
                owned = self.owned[unit_id]
                if owned > owns:
                    self.temp_id[unit_id] += 1
            unit_pos = own_units[unit_id]
            home_force = repelling_force(unit_pos, self.params.home_base)
            closest_enemy_d = self.closest_enemy_dist(update, unit_id, own_units)

            if closest_enemy_d < 2:  # RUN AWAY TO HOME

                force = to_polar(normalize((home_force * HOME_INFLUENCE)))
                move = (force[0], force[1] + np.pi)
                moves[unit_id] = move
            elif closest_enemy_d < 5:  # STAY PUT
                moves[unit_id] = (0, 0)
            else:
                ux, uy = unit_pos
                if self.params.player_idx == 0:
                    wall_normals = [
                        (ux, self.params.min_dim),
                        (self.params.min_dim, uy),
                    ]
                elif self.params.player_idx == 1:
                    wall_normals = [
                        (ux, self.params.max_dim),
                        (self.params.min_dim, uy),
                    ]
                elif self.params.player_idx == 2:
                    wall_normals = [
                        (ux, self.params.max_dim),
                        (self.params.max_dim, uy),
                    ]
                elif self.params.player_idx == 3:
                    wall_normals = [
                        (ux, self.params.min_dim),
                        (self.params.max_dim, uy),
                    ]
                else:
                    pass

                horizontal_influence = np.random.randint(40) - 10
                if self.temp_id[unit_id] % 2 == 0:
                    horizontal_force = repelling_force(unit_pos, wall_normals[0])
                else:
                    horizontal_force = repelling_force(unit_pos, wall_normals[1])

                if self.temp_id[unit_id] % 4 <= 1:
                    horizontal_influence = np.random.randint(30) - 10

                total_force = normalize(
                    (home_force * HOME_INFLUENCE)
                    + (horizontal_force * horizontal_influence)
                )
                # self._logger.debug("force", total_force)
                moves[unit_id] = to_polar(total_force)
                # self._debug(moves)

        return moves

    def deallocation_candidate(self, update, target_point):
        pass


def check_border(my_player_idx, target_player_idx, vertical, horizontal):
    # pdb.set_trace()
    def check_pair(idx1, idx2, target1, target2):
        if idx1 == target1 and idx2 == target2 or idx1 == target2 and idx2 == target1:
            return True
        else:
            return False

    if check_pair(my_player_idx, target_player_idx, 0, 1):
        return 1 in vertical or 1 in horizontal
    if check_pair(my_player_idx, target_player_idx, 0, 2):
        return 3 in vertical or 3 in horizontal
    elif check_pair(my_player_idx, target_player_idx, 0, 3):
        return 7 in vertical or 7 in horizontal
    elif check_pair(my_player_idx, target_player_idx, 1, 2):
        return 2 in vertical or 2 in horizontal
    elif check_pair(my_player_idx, target_player_idx, 1, 3):
        return 6 in vertical or 6 in horizontal
    elif check_pair(my_player_idx, target_player_idx, 2, 3):
        return 4 in vertical or 4 in horizontal
    else:
        # somethings wrong
        return None


def border_detect(map_state, my_player_idx, target_player_idx):
    base_kernel = torch.tensor([-1, 0, 1])
    vecrtical_kernel = base_kernel.reshape(1, 1, 3, 1)
    horizontal_kernel = base_kernel.reshape(1, 1, 1, 3)
    map_tensor = torch.tensor(map_state).reshape(1, 100, 100)
    # SET OCCUPATION TO SPECIFC VALUE, for conv purposes
    map_tensor[map_tensor == -1] = -10000
    map_tensor[map_tensor == 1] = 1
    map_tensor[map_tensor == 2] = 2
    map_tensor[map_tensor == 4] = 8
    map_tensor[map_tensor == 3] = 4
    # tmp = map_tensor.unique()
    # pdb.set_trace()
    vertical_edge = torch.abs(F.conv2d(map_tensor, vecrtical_kernel))
    horizontal_edge = torch.abs(F.conv2d(map_tensor, horizontal_kernel))
    # pdb.set_trace()
    # plt.imshow(vertical_edge.permute(1,2,0))
    # plt.savefig("vertical.png")
    # plt.imshow(horizontal_edge.permute(1,2,0))
    # plt.savefig("horizontal.png")
    return check_border(
        my_player_idx, target_player_idx, vertical_edge, horizontal_edge
    )


class Attacker(Role):
    def __init__(self, logger, params, target_player):
        super().__init__(logger, params)
        self.pincer_force = dict()
        self.pincer_balance_assignment = 0
        self.target_player = target_player

    def get_centroid(self, units):
        """
        Find centroid on a cluster of points
        """
        return units.mean(axis=0)

    def find_closest_point(self, line, point):
        """
        Find closest point on line segment given a point
        """
        return nearest_points(line, point)[0]

    def _turn_moves(self, update, dead_units):
        # pdb.set_trace()
        # tmp1 = self.params.player_idx
        # tmp2 = self.target_player
        # border_exist = border_detect(update.map_states, self.params.player_idx, self.target_player)
        # pdb.set_trace()

        ATTACK_INFLUENCE = 100
        AVOID_INFLUENCE = 300
        SPREAD_INFLUENCE = 30

        moves = {}
        own_units = update.own_units()
        enemy_units = update.enemy_units()
        enemy_units = enemy_units[self.target_player]

        # get attack force
        homebase_mode = True
        # get where to initiate the attack
        units_array = np.stack([v for k, v in own_units.items()])
        if homebase_mode:  # attack from homebase
            start_point = self.params.spawn_point
        else:
            start_point = self.get_centroid(units_array)
        # get attack target and get formation
        avoid, target = self.find_target_simple(start_point, enemy_units)
        formation = LineString([start_point, target])
        # calcualte force
        for unit_id in self.units:
            unit_pos = own_units[unit_id]
            closest_pt_on_formation = self.find_closest_point(
                formation, Point(unit_pos)
            )

            attack_force = self.attack_point(unit_pos, target, closest_pt_on_formation)

            attack_repulsion_force = repelling_force(unit_pos, avoid)

            attack_unit_spread_force = self.attacker_spread_force(
                unit_pos, unit_id, own_units
            )

            if unit_id not in self.pincer_force:
                pincer_spread_force = self.pincer_spread_force(attack_force)
                self.pincer_force[unit_id] = pincer_spread_force

            dist_to_avoid = np.linalg.norm(avoid - unit_pos)
            PINCER_INFLUENCE = 1 / dist_to_avoid * 30

            total_force = normalize(
                SPREAD_INFLUENCE * attack_unit_spread_force
                + +AVOID_INFLUENCE * attack_repulsion_force
                + ATTACK_INFLUENCE * attack_force
                + PINCER_INFLUENCE * self.pincer_force[unit_id]
            )

            moves[unit_id] = to_polar(total_force)
        return moves

    def find_target_simple(self, start_point, enemy_units):
        # find a target to attack
        # return 2 points, point A is where the enemy is
        # point B is where we wanna go
        # point A create a strong repulsive force
        # Point B create a strong attracking force
        # In hope that when going toward point B, our attack would "circle around" point A
        # For now, just find a, and use heuristic(fix distance) for b
        # find the closest point to our base
        dist_to_home = []
        for _, enemy_pos in enemy_units:
            dist_to_home.append(
                np.linalg.norm(self.params.home_base - enemy_pos).item()
            )
        target_idx = np.argmin(np.array(dist_to_home))
        target_pos = enemy_units[target_idx][1]
        unit_vec, _ = force_vec(start_point, target_pos)
        return target_pos, target_pos - unit_vec * 10

    def attacker_spread_force(self, my_pos, my_id, own_units):
        # pdb.set_trace()
        attacker_unit_forces = [
            repelling_force(my_pos, ally_pos)
            for unit_id, ally_pos in own_units.items()
            if unit_id != my_id
        ]
        spread_force = np.add.reduce(attacker_unit_forces)
        return normalize(spread_force)

    def pincer_spread_force(self, attack_force):
        # alteranting left and right of the target point
        vec_perpendicular = attack_force.copy()
        vec_perpendicular[self.pincer_balance_assignment] *= -1
        self.pincer_balance_assignment = int(not bool(self.pincer_balance_assignment))
        return vec_perpendicular

    def attack_point(self, unit, target, closest_point):
        """Given a unit, attack the target point following the foramtion.
        Args:
            units: attack unit
            target: attack target
            cloest_points: cloest point a unit is from its formation
        Return:
            attack vector following the formation
        """
        unit_vec_closest, mag_closest = force_vec(unit, closest_point.coords)
        unit_vec_target, mag_target = force_vec(unit, target)
        # Calculate weight for cloest point and target point
        total_mag = mag_target + mag_closest + EPSILON
        weight_target = mag_target / total_mag
        weight_closest = mag_closest / total_mag
        # Calculate move vec for each units
        attack_vec = unit_vec_closest * weight_closest + unit_vec_target * weight_target
        attack_vec = attack_vec
        attack_vec *= -1
        return attack_vec[0]

    def deallocation_candidate(self, update, target_point):
        """
        distance = []
        for unit_id in self.units:
            unit_pos = own_units[unit_id]
            _, dist = force_vec(target_point, unit_pos)
        """
        pass


# =======================================
# Dynamic Allocation Rules Infrastructure
# =======================================
class RoleGroups:
    __role_groups: dict[RoleType, list[Role]]
    __id_counters: dict[RoleType, int]

    def __init__(self):
        self.__role_groups = {role: [] for role in RoleType}
        self.__id_counters = {role: 0 for role in RoleType}

    def all_roles(self) -> list[Role]:
        return [role for group in self.__role_groups.values() for role in group]

    def of_type(self, type: RoleType) -> list[Role]:
        return self.__role_groups[type]

    def add_group(self, type: RoleType, role: Role):
        role.id = f"{type}::{self.__id_counters[type]}"
        self.__id_counters[type] += 1
        self.__role_groups[type].append(role)

    def remove_group(self, type: RoleType, id: str):
        self.__role_groups[type] = [
            role for role in self.__role_groups[type] if role.id != id
        ]


class RuleInputs:
    logger: logging.Logger
    params: GameParameters
    update: StateUpdate
    role_groups: RoleGroups

    def __init__(
        self,
        logger: logging.Logger,
        params: GameParameters,
        update: StateUpdate,
        role_groups: RoleGroups,
    ):
        self.logger = logger
        self.params = params
        self.update = update
        self.role_groups = role_groups

    def debug(self, *args):
        self.logger.info(" ".join(str(a) for a in args))


class SpawnRules:
    RuleFunc: TypeAlias = Callable[[RuleInputs, str], bool]
    """
    A Rule application function.

    Return `True` if we should stop processing more rules, or `False` otherwise.
    """

    rules: list[RuleFunc]

    def __init__(self):
        self.rules = []

    def rule(self, rule_func: RuleFunc):
        self.rules.append(rule_func)

    def apply_rules(self, rule_inputs: RuleInputs, uid: str):
        for rule_func in self.rules:
            if rule_func(rule_inputs, uid):
                return
        rule_inputs.debug("Failed to apply any rules to", uid)


class ReallocationRules:
    RuleFunc: TypeAlias = Callable[[RuleInputs], bool]
    """
    A Rule application function.

    Return `True` if this rule did something, `False` otherwise.
    """

    MAX_ROUNDS = 10

    rules: list[RuleFunc]

    def __init__(self):
        self.rules = []

    def rule(self, rule_func: RuleFunc):
        self.rules.append(rule_func)

    def apply_rules(self, rule_inputs: RuleInputs):
        continuing = True
        rounds = 0
        while continuing and rounds < self.MAX_ROUNDS:
            rounds += 1
            continuing = False
            for rule_func in self.rules:
                if rule_func(rule_inputs):
                    continuing = True


# =============================
# Spawned Unit Allocation Rules
# =============================
# Order of rules declared below is their precedence
# Higher up means higher precedence
spawn_rules = SpawnRules()


@spawn_rules.rule
def early_scouts(rule_inputs: RuleInputs, uid: str):
    if rule_inputs.update.turn < 20:
        rule_inputs.role_groups.of_type(RoleType.SCOUT)[0].allocate_unit(uid)
        return True
    return False


@spawn_rules.rule
def populate_defenders(rule_inputs: RuleInputs, uid: str):
    defender_role = rule_inputs.role_groups.of_type(RoleType.DEFENDER)[0]
    if rule_inputs.update.turn > 30 and len(defender_role.units) < 40:
        defender_role.allocate_unit(uid)
        return True
    return False


@spawn_rules.rule
def all_scouts(rule_inputs: RuleInputs, uid: str):
    rule_inputs.role_groups.of_type(RoleType.SCOUT)[0].allocate_unit(uid)
    return True


# Disabled for class
# @spawn_rules.rule
def even_scouts_attackers(rule_inputs: RuleInputs, uid: str):
    scout_roles = rule_inputs.role_groups.of_type(RoleType.SCOUT)
    attacker_roles = rule_inputs.role_groups.of_type(RoleType.ATTACKER)
    total_scouts = sum(len(scouts.units) for scouts in scout_roles)
    total_attackers = sum(len(attackers.units) for attackers in attacker_roles)
    if total_scouts >= total_attackers:
        global attack_roll
        attacker_roles[attack_roll % 3].allocate_unit(uid)
        attack_roll += 1
    else:
        scout_roles[0].allocate_unit(uid)
    return True


# =======================
# Role Reallocation Rules
# =======================
# Order of rules declared below is their precedence
# Higher up means higher precedence
reallocation_rules = ReallocationRules()


@reallocation_rules.rule
def reassign_defenders_on_sufficient_density(rule_inputs: RuleInputs):
    return False


@reallocation_rules.rule
def form_interceptors_on_threat(rule_inputs: RuleInputs):
    # Visualize Clusters
    # if rule_inputs.params.player_idx == 0 and rule_inputs.update.turn % 10 == 0:
    #     player_unit_id_to_pos = {
    #         (player, uid): pos
    #         for player in range(4)
    #         for uid, pos in zip(
    #             rule_inputs.update.unit_id[player], rule_inputs.update.unit_pos[player]
    #         )
    #     }
    #     plt.clf()
    #     clusters = rule_inputs.update.enemy_clusters()
    #     for enemy, enemy_clusters in clusters.items():
    #         for cluster, cluster_units in enemy_clusters.items():
    #             cluster_positions = [
    #                 player_unit_id_to_pos[(enemy, unit_id)] for unit_id in cluster_units
    #             ]
    #             xs = [x for x, _ in cluster_positions]
    #             ys = [y for _, y in cluster_positions]
    #             plt.plot(xs, ys, marker="o")
    #     plt.savefig(f"debug/{rule_inputs.update.turn}.png")
    per_enemy_clusters = rule_inputs.update.enemy_clusters()
    all_clusters = [
        (
            enemy,
            [uid for uid in cluster_units],
            [rule_inputs.update.unit_id_to_pos(enemy, uid) for uid in cluster_units],
        )
        for enemy in rule_inputs.update.enemies()
        for _, cluster_units in per_enemy_clusters[enemy].items()
    ]
    for enemy in rule_inputs.update.enemies():
        for cluster_units in per_enemy_clusters[enemy].values():
            if any(
                (enemy, cluster_unit) in interceptor.targets
                for cluster_unit in cluster_units
                for interceptor in rule_inputs.role_groups.of_type(RoleType.INTERCEPTOR)
            ):
                continue
            target_points = MultiPoint(
                [
                    Point(rule_inputs.update.unit_id_to_pos(enemy, uid))
                    for uid in cluster_units
                ]
            )
            nearest_target, _ = nearest_points(
                target_points, Point(rule_inputs.params.home_base)
            )
            nearest_target = point_to_floats(nearest_target)
            # target_center = point_to_floats(target_points.centroid)
            if np.linalg.norm(rule_inputs.params.home_base - nearest_target) < 50:
                rule_inputs.debug(
                    f"Triggered interceptors: {len(cluster_units)} targets"
                )
                interceptor_role = Interceptor(
                    rule_inputs.logger, rule_inputs.params, enemy, cluster_units
                )
                rule_inputs.role_groups.add_group(
                    RoleType.INTERCEPTOR, interceptor_role
                )
                defender_role = rule_inputs.role_groups.of_type(RoleType.DEFENDER)[0]
                # TODO: Magic number for how many interceptors to allocate
                for _ in range(20):
                    if len(defender_role.units) == 0:
                        break
                    converted_defender = defender_role.deallocation_candidate(
                        rule_inputs.update, nearest_target
                    )
                    defender_role.deallocate_unit(converted_defender)
                    interceptor_role.allocate_unit(converted_defender)
                return True
    return False


@reallocation_rules.rule
def reinforce_interceptors(rule_inputs: RuleInputs):
    return False


@reallocation_rules.rule
def reinforce_defenders(rule_inputs: RuleInputs):
    return False


@reallocation_rules.rule
def disband_interceptors_target_gone(rule_inputs: RuleInputs):
    for interceptors in rule_inputs.role_groups.of_type(RoleType.INTERCEPTOR):
        if len(interceptors.targets) == 0:
            rule_inputs.debug("Removing interceptor!")
            for uid in interceptors.units:
                rule_inputs.role_groups.of_type(RoleType.DEFENDER)[0].allocate_unit(uid)
            rule_inputs.role_groups.remove_group(RoleType.INTERCEPTOR, interceptors.id)
    return False


# ======
# Player
# ======
class Player:
    params: GameParameters
    logger: logging.Logger
    turn: int
    known_units: set[str]
    role_groups: RoleGroups

    def __init__(
        self,
        rng: np.random.Generator,
        logger: logging.Logger,
        total_days: int,
        spawn_days: int,
        player_idx: int,
        spawn_point: Point,
        min_dim: int,
        max_dim: int,
        precomp_dir: str,
    ) -> None:
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
        self.player_idx = player_idx
        self.turn = 0
        self.known_units = set()

        # Game fundamentals
        self.params = GameParameters()
        self.params.total_days = total_days
        self.params.spawn_days = spawn_days
        self.params.player_idx = player_idx
        self.params.spawn_point = (float(spawn_point.x), float(spawn_point.y))
        self.params.min_dim = min_dim
        self.params.max_dim = max_dim
        self.params.home_base = [(-1, -1), (-1, 101), (101, 101), (101, -1)][player_idx]

        self.role_groups = RoleGroups()
        self.role_groups.add_group(
            RoleType.DEFENDER, LatticeDefender(self.logger, self.params, 40)
        )
        enemy_player = set([0, 1, 2, 3]) - set([player_idx])
        for enemy_idx in enemy_player:
            self.role_groups.add_group(
                RoleType.ATTACKER, Attacker(self.logger, self.params, enemy_idx)
            )
        self.role_groups.add_group(
            RoleType.SCOUT, GreedyScout(self.logger, self.params)
        )

    def debug(self, *args):
        self.logger.info(" ".join(str(a) for a in args))

    def risk_distances(self, enemy_location, own_units):
        d_base = np.linalg.norm(np.subtract(self.params.home_base, enemy_location))
        d_to_closest_unit = 150
        for unit in own_units:
            d_our_unit = np.linalg.norm(np.subtract(unit[1], enemy_location))
            d_to_closest_unit = min(d_our_unit, d_to_closest_unit)
        return (d_base, d_to_closest_unit)

    def play(
        self, unit_id, unit_pos, map_states, current_scores, total_scores
    ) -> list[tuple[float, float]]:
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

        # Convert unit positions to floats
        unit_pos = [
            [point_to_floats(player_unit_pos) for player_unit_pos in player_units]
            for player_units in unit_pos
        ]

        update = StateUpdate(self.params, self.turn, unit_id, unit_pos, map_states)

        own_units = list(update.own_units().items())
        enemy_unit_locations = [pos for _, pos in update.all_enemy_units()]
        risk_distances = [
            self.risk_distances(enemy_location, own_units)
            for enemy_location in enemy_unit_locations
        ]

        risks = list(
            zip(
                enemy_unit_locations,
                [
                    min(100, (750 / (d1 + EPSILON) + 750 / (d2 + EPSILON)))
                    for d1, d2 in risk_distances
                ],
            )
        )
        # visualize_risk(risks, enemy_unit_locations, own_units, self.turn)

        # Calculate free units (just spawned)
        own_units = set(uid for uid in unit_id[self.params.player_idx])
        allocated_units = set(
            uid for role in self.role_groups.all_roles() for uid in role.units
        )
        free_units = own_units - allocated_units
        just_spawned = set(uid for uid in free_units if not uid in self.known_units)
        idle_units = free_units - just_spawned
        if len(idle_units) > 0:
            self.debug(f"{len(idle_units)} idle units!")

        # Phase 1: apply spawn rules
        # --------------------------
        rule_inputs = RuleInputs(self.logger, self.params, update, self.role_groups)
        for uid in just_spawned:
            spawn_rules.apply_rules(rule_inputs, uid)

        # Phase 2: apply reallocation rules
        # ---------------------------------
        reallocation_rules.apply_rules(rule_inputs)

        # Phase 3: gather moves from roles
        # --------------------------------
        moves: list[tuple[float, float]] = []
        role_moves = {}
        for role in RoleType:
            for role_group in self.role_groups.of_type(role):
                if len(role_group.units) > 0:
                    try:
                        role_moves.update(role_group.turn_moves(update))
                    except Exception as e:
                        self.debug(
                            "Exception processing role moves:", traceback.format_exc()
                        )
        for unit_id in unit_id[self.params.player_idx]:
            if not unit_id in role_moves:
                moves.append((0, 0))
            else:
                moves.append(role_moves[unit_id])

        self.turn += 1
        return moves


def visualize_risk(risks, enemy_units_locations, own_units, turn):
    DISPLAY_EVERY_N_ROUNDS = 30
    HEAT_MAP = False

    if HEAT_MAP and turn % DISPLAY_EVERY_N_ROUNDS == 0:

        c = []
        for r in risks:
            c.append(r[1])
        plt.rcParams["figure.autolayout"] = True
        x = np.array(enemy_units_locations)[:, 0]
        y = np.array(enemy_units_locations)[:, 1]
        c = np.array(c)

        df = pd.DataFrame({"x": x, "y": y, "c": c})

        fig, ax = plt.subplots()  # 1,1, figsize=(20,6))
        cmap = plt.cm.hot
        norm = colors.Normalize(vmin=0.0, vmax=100.0)
        mp = ax.scatter(df.x, df.y, color=cmap(norm(df.c.values)))
        ax.set_xticks(df.x)

        fig.subplots_adjust(right=0.9)
        sub_ax = plt.axes([0.8, 0.4, 0.1, 0.4])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, cax=sub_ax)
        ax.invert_yaxis()

        for p in range(1):
            for num, pos in own_units:
                ax.scatter(pos[0], pos[1], color="blue")

        np.meshgrid(list(range(100)), list(range(100)))
        plt.title(f"Day {turn}")
        plt.tight_layout()
        plt.savefig(f"risk_{turn}.png")


unit_colors = {}


def visualize_ownership(turn: int, update: StateUpdate):
    plt.clf()
    ax = plt.gca()

    cmap = colors.ListedColormap([dispute_color] + tile_color)
    bounds = [-1, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    X, Y = np.meshgrid(list(range(100)), list(range(100)))
    plt.pcolormesh(
        X + 0.5,
        Y + 0.5,
        np.transpose(update.map_states),
        cmap=cmap,
        norm=norm,
    )

    # Units
    for p in range(4):
        for x, y in update.unit_pos[p]:
            plt.plot(
                x,
                y,
                color=player_color[p],
                marker="o",
                markersize=4,
                markeredgecolor="black",
            )

    # Ownership
    unit_to_owned, _ = update.unit_ownership()
    lines = {}
    line_colors = {}
    for player in range(4):
        lines[player] = []
        line_colors[player] = []
        for unit_id, owned in unit_to_owned[player].items():
            if not (player, unit_id) in unit_colors:
                unit_colors[(player, unit_id)] = (
                    randint(0, 255),
                    randint(0, 255),
                    randint(0, 255),
                )
            unit_idx = update.unit_id[player].index(unit_id)
            unit_x, unit_y = update.unit_pos[player][unit_idx]
            for tile_x, tile_y in owned:
                lines[player].append(
                    [
                        (int(unit_x) + 0.5, int(unit_y) + 0.5),
                        (tile_x + 0.5, tile_y + 0.5),
                    ]
                )
                line_colors[player].append(unit_colors[(player, unit_id)])
    for player in range(4):
        plt.gca().add_collection(
            LineCollection(
                lines[player], linewidth=1, alpha=0.2, colors=line_colors[player]
            )
        )

    plt.xticks(np.arange(0, 100, 10))
    plt.yticks(np.arange(0, 100, 10))
    plt.grid(color="black", alpha=0.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    ax.set_aspect(1)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.invert_yaxis()
    plt.title(f"Day {turn}")
    plt.savefig(f"debug/{turn}.png", dpi=300)


makedirs("debug", exist_ok=True)
existing = glob("debug/*.png")
for f in existing:
    remove(f)
