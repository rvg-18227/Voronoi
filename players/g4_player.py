import logging
import math
import multiprocessing
import pdb
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from glob import glob
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

THREADED = False

# ==================
# Game State Classes
# ==================

attack_roll = 0


class GameParameters:
    """Represents constant game parameters that don't change."""

    total_days: int
    spawn_days: int
    player_idx: int
    spawn_point: NDArray[np.float32]
    min_dim: int
    max_dim: int
    home_base: NDArray[np.float32]


class StrategyParameters:
    min_defenders: int
    interceptor_strength: int

    def __init__(self, params: GameParameters):
        self.min_defenders = 20 if params.spawn_days < 5 else 10
        self.interceptor_strength = 3
        if params.spawn_days < 2:
            self.defender_area = 100
        elif params.spawn_days < 5:
            self.defender_area = 200
        else:
            self.defender_area = 500

        if params.spawn_days < 5:
            self.scout_ownership = 10
        else:
            self.scout_ownership = 20


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
if THREADED:
    pool = multiprocessing.Pool(multiprocessing.cpu_count())


def in_bounds(params: GameParameters, x: int, y: int):
    return x >= 0 and y >= 0 and x < params.max_dim and y < params.max_dim


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
    cached_territory_hull: Optional[Polygon]
    cached_territory_poly: Optional[Polygon]

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
        self.cached_territory_hull = None
        self.cached_territory_poly = None

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

    def unit_idx_to_id(self, player: int, idx: int) -> Optional[str]:
        unit_idx_to_id = self.player_unit_idx_to_id[player]
        return unit_idx_to_id[idx] if idx in unit_idx_to_id else None

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
            player: {uid: [] for uid in self.unit_id[player]} for player in range(4)
        }
        # (x, y) -> (player, unit)
        tile_to_unit: dict[tuple[int, int], tuple[int, str]] = {}

        player_kdtrees = {
            player: KDTree(self.unit_pos[player])
            for player in range(4)
            if len(self.unit_pos[player]) > 0
        }
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
                        self.player_unit_idx_to_id[owning_player],
                    )
                )

        if THREADED:
            results = pool.map(get_nearest_unit, work)
        else:
            results = map(get_nearest_unit, work)

        for result in results:
            pos, owning_player, closest_uid = result
            unit_to_owned[owning_player][closest_uid].append(pos)
            tile_to_unit[pos] = (owning_player, closest_uid)

        self.cached_ownership = (unit_to_owned, tile_to_unit)
        return self.cached_ownership

    def enemy_clusters(self):
        if self.cached_clusters is not None:
            return self.cached_clusters

        enemies = self.enemies()
        enemy_units = self.enemy_units()
        dbscans = {
            enemy: DBSCAN(eps=5, min_samples=1).fit(
                [unit_pos for _, unit_pos in enemy_units[enemy]]
            )
            for enemy in enemies
            if len(enemy_units[enemy]) > 0
        }
        clusters: dict[int, dict[int, list[str]]] = {enemy: {} for enemy in enemies}
        for enemy in enemies:
            if not enemy in dbscans:
                continue
            for unit_idx, label in enumerate(dbscans[enemy].labels_):
                if label == -1:
                    continue
                if not label in clusters[enemy]:
                    clusters[enemy][label] = []
                clusters[enemy][label].append(
                    self.player_unit_idx_to_id[enemy][unit_idx]
                )
        self.cached_clusters = clusters
        return clusters

    def territory_hull(self):
        if self.cached_territory_hull is not None:
            return self.cached_territory_hull

        hull_poly = self.territory_poly().convex_hull
        envelope = box(0, 0, self.params.max_dim, self.params.max_dim)
        self.cached_territory_hull = hull_poly.union(hull_poly.buffer(1)).intersection(
            envelope
        )
        return self.cached_territory_hull

    def territory_poly(self):
        if self.cached_territory_poly is not None:
            return self.cached_territory_poly

        move = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        vertex_off = [(0, 0), (0, 1), (1, 1), (1, 0)]
        direction = 0
        spawn_x, spawn_y = map(int, self.params.spawn_point)
        vertices = []
        x, y = spawn_x, spawn_y
        own_territory = [
            [cell == self.params.player_idx + 1 for cell in col]
            for col in self.map_states
        ]

        while len(vertices) < 4 or (x, y) != (spawn_x, spawn_y):
            vx, vy = vertex_off[direction]
            vertices.append((x + vx, y + vy))

            try_right_dir = (direction - 1) % 4
            dx, dy = move[try_right_dir]
            try_right_x = x + dx
            try_right_y = y + dy

            dx, dy = move[direction]
            continue_x = x + dx
            continue_y = y + dy
            # Try to turn right
            if (
                in_bounds(self.params, try_right_x, try_right_y)
                and own_territory[try_right_x][try_right_y]
            ):
                direction = try_right_dir
                x = try_right_x
                y = try_right_y
            # Continue in same direction
            elif (
                in_bounds(self.params, continue_x, continue_y)
                and own_territory[continue_x][continue_y]
            ):
                x = continue_x
                y = continue_y
                continue
            # Turn left
            else:
                direction = (direction + 1) % 4

        self.cached_territory_poly = Polygon(vertices)
        return self.cached_territory_poly


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


def to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


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


def near(a, b):
    return np.abs(a - b) < 0.01


def retreat_force(
    update: StateUpdate, unit_pos: NDArray[np.float32], retreat_threshold: float
):
    territory_boundary = update.territory_poly().exterior
    longest_ray_length = 0
    longest_ray_theta = 0
    total_rays = 36
    short_rays = 0
    pos_point = Point(unit_pos)
    # Ray casting
    for theta in np.linspace(0, 2 * np.pi, 36):
        dx = 50 * np.cos(theta)
        dy = 50 * np.sin(theta)
        ux, uy = unit_pos
        ray = LineString([(ux, uy), (ux + dx, uy + dy)])
        strike = ray.intersection(territory_boundary)
        if strike.is_empty:
            longest_ray_length = 50
            longest_ray_theta = theta
        else:
            ray_length = pos_point.distance(strike)
            if ray_length > longest_ray_length:
                longest_ray_length = ray_length
                longest_ray_theta = theta

            if ray_length < 20:
                short_rays += 1

    if (short_rays / total_rays) > retreat_threshold:
        # Retreat direction of longest ray
        return np.array(to_cartesian(1, longest_ray_theta))
    else:
        return np.array([0, 0])


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
    __logger: logging.Logger
    _params: GameParameters
    __allocated_units: list[str]

    def __init__(self, logger, params):
        self.__logger = logger
        self._params = params

        self.id = ""
        self.__allocated_units = []

    def debug(self, *args):
        self.__logger.info(" ".join(str(a) for a in args))

    # ===========
    # Role Common
    # ===========

    def allocate_unit(self, unit_id: str):
        self.__allocated_units.append(unit_id)

    def deallocate_unit(self, unit_id: str):
        if not unit_id in self.__allocated_units:
            raise KeyError(f"Unit {unit_id} is not in {self.id}")
        self.__allocated_units = [
            uid for uid in self.__allocated_units if uid != unit_id
        ]

    @property
    def units(self):
        return self.__allocated_units.copy()

    @property
    def params(self):
        return self._params

    def turn_update(self, update: StateUpdate, **kwargs):
        alive_units = set(update.own_units().keys())
        allocated_set = set(self.__allocated_units)
        dead_units = allocated_set - alive_units
        self.__allocated_units = list(allocated_set - dead_units)

        self._turn_update(update, dead_units)

    def turn_moves(self, update: StateUpdate, **kwargs):
        return self._turn_moves(update, **kwargs)

    def _closest_unit(self, update: StateUpdate, target_point: NDArray[np.float32]):
        unit_idx_to_id = {idx: uid for idx, uid in enumerate(self.units)}
        kdtree = KDTree(
            [update.unit_id_to_pos(self.params.player_idx, uid) for uid in self.units]
        )
        _, unit_idx = kdtree.query(target_point)
        return unit_idx_to_id[unit_idx]

    # ===================
    # Role Specialization
    # ===================

    @abstractmethod
    def _turn_update(self, update: StateUpdate, dead_units: set[str]):
        """Initial pass state update: Remove dead units, make any internal state adjustments before processing rules."""
        pass

    @abstractmethod
    def _turn_moves(self, update: StateUpdate) -> dict[str, NDArray[np.float32]]:
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
    counter: int

    def __init__(self, logger, params):
        super().__init__(logger, params)
        self.__spawn_jitter = {}
        self.counter = 0

    def inside_territory(self, update: StateUpdate, unit_pos: NDArray[np.float32]):
        pos_point = Point(unit_pos)
        territory = update.territory_hull()
        in_vec, _ = force_vec(self.params.home_base, unit_pos)
        if not pos_point.intersects(territory):
            return in_vec
        else:
            _, nearest_border = nearest_points(pos_point, territory.exterior)
            border_distance = pos_point.distance(territory.exterior)
            # Near the walls doesn't count
            if (
                near(nearest_border.x, 0)
                or near(nearest_border.x, self.params.max_dim)
                or near(nearest_border.y, 0)
                or near(nearest_border.y, self.params.max_dim)
            ):
                return np.array([0, 0])
            # On the interior of the territory
            elif border_distance > 10:
                return np.array([0, 0])
            # Near the border of the territory
            else:
                return in_vec

    def get_jitter(self, uid: str):
        # Add decaying random direction bias
        if not uid in self.__spawn_jitter:
            self.__spawn_jitter[uid] = np.array((uniform(0.1, 1), uniform(0, 0.9)))

        jitter_force = self.__spawn_jitter[uid]
        self.__spawn_jitter[uid] *= 0.5

        return jitter_force

    def _turn_update(self, update, dead_units):
        pass

    def _turn_moves(self, update):
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

            defender_positions = {
                id: pos
                for id, pos in zip(
                    update.unit_id[self.params.player_idx],
                    update.unit_pos[self.params.player_idx],
                )
                if id in self.units
            }

            moves = {}
            for uid, pos in defender_positions.items():
                jitter_force = self.get_jitter(uid)

                unit_point = Point(pos[0], pos[1])
                target = pos
                for poly in fixed_voronoi_polys:
                    if poly.contains(unit_point):
                        target = np.array([poly.centroid.x, poly.centroid.y])
                        continue
                moves[uid] = to_polar(
                    normalize(
                        jitter_force
                        + normalize(target - pos)
                        + (100 * self.inside_territory(update, pos))
                    )
                )
            return moves
        except Exception as e:
            self.debug("Exception when processing LatticeDefender moves", e)
            return {
                uid: to_polar(normalize(self.get_jitter(uid))) for uid in self.units
            }

    def deallocation_candidate(self, update, target_point):
        return self._closest_unit(update, target_point)


class Interceptor(Role):
    __target_player: int
    __targets: set[str]

    def __init__(self, logger, params, target_player: int, targets: list[str]):
        super().__init__(logger, params)

        self.__target_player = target_player
        self.__targets = set(targets)

        self.noise = dict()

    @property
    def target_player(self):
        return self.__target_player

    @property
    def target_units(self):
        return self.__targets

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

    def _turn_update(self, update, dead_units):
        target_enemy_units = set(update.unit_id[self.__target_player])
        # Prune dead targets
        self.__targets = self.__targets.intersection(target_enemy_units)

    def _turn_moves(self, update):
        # Should never happen due to empty targets Interceptor removal rule
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

            dist_to_avoid = np.linalg.norm(avoid - unit_pos)
            noise_influence = 1 / dist_to_avoid * 10

            total_force = normalize(
                attack_repulsion_force * AVOID_INFLUENCE
                + ATTACK_INFLUENCE * attack_force
                + noise_influence * self.noise[unit_id]
                # + 100 * retreat_force(update, unit_pos, 0.8)
            )

            moves[unit_id] = to_polar(total_force)
        return moves

    def noise_force(self, attack_force):
        # generate noise force roughly in the same direction as attack force
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


class GreedyScout(Role):
    def __init__(self, logger, params):
        super().__init__(logger, params)
        self.owned = {}
        self.first_turn = True

        # unit_id -> scout_id
        self.angles = {}
        self.temp_id = {}
        self.id_counter = 0

    def closest_enemy_dist(self, update, uid, own_units):
        enemy_units = update.all_enemy_units()
        p = own_units[uid]
        closest_dist = 500
        for _, enemy_pos in enemy_units:
            closest_dist = min(closest_dist, np.linalg.norm(p - enemy_pos))
        return closest_dist

    def _turn_update(self, update, dead_units):
        pass

    def _turn_moves(self, update):
        for uid in self.units:
            if not uid in self.temp_id:
                self.temp_id[uid] = self.id_counter
                self.id_counter += 1
                self.angles[uid] = -1

        HOME_INFLUENCE = 30
        own_units = update.own_units()
        moves = {}
        unit_to_owned, _ = update.unit_ownership()
        presets = [15, 75, 30, 60, 45]
        for unit_id in self.units:
            owns = 0
            if unit_id in unit_to_owned[self.params.player_idx]:
                owns = len(unit_to_owned[self.params.player_idx][unit_id])
            if not unit_id in self.owned:
                self.owned[unit_id] = owns
                owned = owns
            else:
                owned = self.owned[unit_id]

            unit_pos = own_units[unit_id]
            home_force = repelling_force(unit_pos, self.params.home_base)
            closest_enemy_d = self.closest_enemy_dist(update, unit_id, own_units)

            retreat = retreat_force(update, unit_pos, 0.8)
            if np.linalg.norm(retreat) > 0:
                moves[unit_id] = to_polar(normalize(retreat))
                continue

            if closest_enemy_d < 2:  # RUN AWAY TO HOME
                force = to_polar(normalize((home_force * HOME_INFLUENCE)))
                move = (1, force[1] + np.pi)
                moves[unit_id] = move
            elif closest_enemy_d < 5:  # STAY PUT
                moves[unit_id] = (0, 0)
            else:
                if owned > owns:
                    self.angles[unit_id] = -1
                if self.angles[unit_id] == -1:
                    if self.params.player_idx == 0:
                        offset = 0
                    elif self.params.player_idx == 1:
                        offset = 270
                    elif self.params.player_idx == 2:
                        offset = 180
                    elif self.params.player_idx == 3:
                        offset = 90
                    if self.temp_id[unit_id] < 5 and owned <= owns:
                        angle = np.radians(presets[self.temp_id[unit_id]] + offset)
                    else:
                        angle = np.radians(np.random.randint(90) + offset)

                    self.angles[unit_id] = (1, angle)
                    moves[unit_id] = (1, angle)
                else:
                    d, a = self.angles[unit_id]
                    moves[unit_id] = (d, a)

        return moves

    def deallocation_candidate(self, update, target_point):
        closest_dist = 500
        closest_uid = 0
        own_units = update.own_units()
        for uid in self.units:
            unit_pos = own_units[uid]
            dist = np.linalg.norm(target_point - unit_pos)
            if dist < closest_dist:
                closest_uid = uid
                closest_dist = dist
        return closest_uid


# Attacker stuff==============================================================
def check_border(my_player_idx, target_player_idx, vertical, horizontal):
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
    vertical_edge = torch.abs(F.conv2d(map_tensor, vecrtical_kernel))
    horizontal_edge = torch.abs(F.conv2d(map_tensor, horizontal_kernel))

    return check_border(
        my_player_idx, target_player_idx, vertical_edge, horizontal_edge
    )


def target_rank(update, start_point, my_player_idx, num_target):
    # return heuristic target with len(targets) = num_targets
    unit_ownership = update.unit_ownership()[0]
    enemy_units = update.enemy_units()
    # for each unit, assign a score based on unit it own and distance from base (risk)
    heuristic_lookup = dict()
    WEIGHT_TILE = 100
    WEIGHT_DIST = 50
    for player_idx in unit_ownership:
        if my_player_idx == player_idx:
            continue
        for unit_id in unit_ownership[player_idx]:
            tile_owned = len(unit_ownership[player_idx][unit_id])
            heuristic_lookup[(player_idx, unit_id)] = (
                tile_owned / 10000 * WEIGHT_TILE
            )  # normalize it with max tile
    for player_idx in enemy_units:
        for unit_id, unit_pos in enemy_units[player_idx]:
            _, distance_to_start = force_vec(unit_pos, start_point)
            try:
                heuristic_lookup[(player_idx, unit_id)] += (
                    1
                    / (distance_to_start / math.sqrt(100**2 + 100**2) + EPSILON)
                    * WEIGHT_DIST
                )  # normalize it with max distaince on board
            except KeyError:
                heuristic_lookup[(player_idx, unit_id)] = -100
                print("somethings wrong")
    # rank the heuristic value
    heuristic_ranking = [(k, v) for k, v in heuristic_lookup.items()]
    heuristic_ranking.sort(key=lambda x: x[1], reverse=True)
    top_k = heuristic_ranking[:num_target]
    targets = []
    for t, h in top_k:
        targets.append(update.unit_id_to_pos(t[0], t[1]))
    # use n unit vec pas target pos as heurist for attacking
    avoids = []
    for target_pos in targets:
        unit_vec, _ = force_vec(start_point, target_pos)
        avoids.append(target_pos - unit_vec * 10)
    return avoids, targets


def density_heatmap(enemy_unit):
    # create a heatmap base on enemy density
    kernel = torch.tensor([[1, 1, 1], [1, 3, 1], [1, 1, 1]], dtype=torch.float)
    kernel = kernel / torch.norm(kernel, p=2)
    kernel = kernel.reshape(1, 1, 3, 3)
    heat_map = torch.zeros(100, 100)
    for _, unit_pos in enemy_unit:
        x, y = unit_pos
        x = int(x)
        y = int(y)
        heat_map[x, y] += 1
    heat_map = heat_map.reshape(1, 100, 100)
    heat_map = F.conv2d(heat_map, kernel, padding="same")
    return heat_map


def assign_target(update, targets, avoids, role_groups, home_base, my_player_idx):
    assignment = []
    role_groups = role_groups.copy()
    for idx, t in enumerate(targets):
        group_dist = defaultdict(list)
        for group in role_groups:
            if len(group.units) == 0:
                _, dist = force_vec(t, home_base)
                group_dist[group].append(dist)
            else:
                for u in group.units:
                    try:
                        u_pos = update.unit_id_to_pos(my_player_idx, u)
                    except KeyError:
                        # dead unit
                        u_pos = home_base
                    _, dist = force_vec(t, u_pos)
                    group_dist[group].append(dist)
        # select the group that has the lowest distance to target
        group_dist = [(k, v) for k, v in group_dist.items()]
        group_dist.sort(key=lambda x: min(x[1]))
        assignment.append((group_dist[0][0], t, avoids[idx]))
        role_groups.remove(group_dist[0][0])
    return assignment


def get_avoid_influence(heatmap, target):
    def clamp(coord):
        coord = int(coord)
        new_coord = coord
        if coord > 99:
            new_coord = 99
        elif coord < 0:
            new_coord = 0
        return new_coord

    base_influence = 300
    x, y = target
    x = clamp(x)
    y = clamp(y)
    density_at_target = heatmap[0, int(x), int(y)].item()
    return density_at_target * base_influence


class Attacker(Role):
    def __init__(self, logger, params):
        super().__init__(logger, params)
        self.pincer_force = dict()
        self.pincer_left_side = False

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

    def _turn_update(self, update, dead_units):
        pass

    def _turn_moves(self, update, avoid, target, avoid_influence):
        ATTACK_INFLUENCE = 200
        AVOID_INFLUENCE = avoid_influence
        SPREAD_INFLUENCE = 100

        moves = {}
        own_units = update.own_units()

        # get attack force
        start_point = self.params.spawn_point
        # get attack target and get formation
        formation = LineString([start_point, target])
        # calcualte force
        for unit_id in self.units:
            unit_pos = own_units[unit_id]
            closest_pt_on_formation = self.find_closest_point(
                formation, Point(unit_pos)
            )

            attack_force = self.attack_point(unit_pos, target, closest_pt_on_formation)

            avoid_repulsion_force = repelling_force(unit_pos, avoid)

            attack_unit_spread_force = self.attacker_spread_force(
                unit_pos, unit_id, own_units
            )

            ux, uy = unit_pos
            wall_normals = [(ux, 0), (ux, 100), (0, uy), (100, uy)]
            wall_forces = [repelling_force(unit_pos, wall) for wall in wall_normals]
            wall_force = normalize(np.add.reduce(wall_forces))

            if unit_id not in self.pincer_force:
                pincer_spread_force = self.pincer_spread_force()
                self.pincer_force[unit_id] = pincer_spread_force

            dist_to_avoid = np.linalg.norm(avoid - unit_pos)
            PINCER_INFLUENCE = AVOID_INFLUENCE / (dist_to_avoid)
            WALL_INFLUENCE = PINCER_INFLUENCE + 30

            total_force = normalize(
                SPREAD_INFLUENCE * attack_unit_spread_force
                + AVOID_INFLUENCE * avoid_repulsion_force
                + ATTACK_INFLUENCE * attack_force
                + PINCER_INFLUENCE
                * normalize(self.pincer_force[unit_id] * attack_force)
                + WALL_INFLUENCE * wall_force
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

    def find_target_sophisticated(self, start_point, enemy_units):
        pass

    def attacker_spread_force(self, my_pos, my_id, own_units):
        attacker_unit_forces = [
            repelling_force(my_pos, ally_pos)
            for unit_id, ally_pos in own_units.items()
            if unit_id != my_id
        ]
        spread_force = np.add.reduce(attacker_unit_forces)
        return normalize(spread_force)

    def pincer_spread_force(self):
        # alteranting left and right of the target point
        dir_vector = None
        if self.pincer_left_side:
            dir_vector = np.array([-1, 1])
            self.pincer_left_side = not self.pincer_left_side
        else:
            dir_vector = np.array([1, -1])
            self.pincer_left_side = not self.pincer_left_side
        return normalize(dir_vector)

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
        return normalize(attack_vec[0])

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
    strategy_params: StrategyParameters
    update: StateUpdate
    role_groups: RoleGroups

    def __init__(
        self,
        logger: logging.Logger,
        params: GameParameters,
        strategy_params: StrategyParameters,
        update: StateUpdate,
        role_groups: RoleGroups,
    ):
        self.logger = logger
        self.params = params
        self.strategy_params = strategy_params
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
            try:
                if rule_func(rule_inputs, uid):
                    return
            except Exception as e:
                rule_inputs.debug(
                    "Exception processing spawn_rules:", traceback.format_exc()
                )
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
                try:
                    if rule_func(rule_inputs):
                        continuing = True
                except Exception as e:
                    rule_inputs.debug(
                        "Exception processing reallocation_rules:",
                        traceback.format_exc(),
                    )


# =============================
# Spawned Unit Allocation Rules
# =============================
# Order of rules declared below is their precedence
# Higher up means higher precedence
spawn_rules = SpawnRules()


@spawn_rules.rule
def early_scouts(rule_inputs: RuleInputs, uid: str):
    if rule_inputs.update.turn < 30:
        rule_inputs.role_groups.of_type(RoleType.SCOUT)[0].allocate_unit(uid)
        return True
    return False


@spawn_rules.rule
def populate_defenders(rule_inputs: RuleInputs, uid: str):
    defender_role = rule_inputs.role_groups.of_type(RoleType.DEFENDER)[0]
    total_interceptors = sum(
        len(interceptor_role.units)
        for interceptor_role in rule_inputs.role_groups.of_type(RoleType.INTERCEPTOR)
    )
    target_density = (
        rule_inputs.update.territory_hull().area
        / rule_inputs.strategy_params.defender_area
    )
    rule_inputs.debug(f"{len(defender_role.units)}/{target_density} defenders")
    if len(defender_role.units) + total_interceptors < target_density:
        defender_role.allocate_unit(uid)
        return True
    return False


@spawn_rules.rule
def low_spawn_no_attackers(rule_inputs: RuleInputs, uid: str):
    if rule_inputs.params.spawn_days > 5:
        scout_role = rule_inputs.role_groups.of_type(RoleType.SCOUT)[0]
        scout_role.allocate_unit(uid)
        return True
    return False


@spawn_rules.rule
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
def reassign_scouts(rule_inputs: RuleInputs):
    if rule_inputs.update.turn < 30:
        return False

    defender_role = rule_inputs.role_groups.of_type(RoleType.DEFENDER)[0]
    scout_role = rule_inputs.role_groups.of_type(RoleType.SCOUT)[0]
    unit_ownership, _ = rule_inputs.update.unit_ownership()
    did_something = False
    for uid in scout_role.units:
        pos_point = Point(
            rule_inputs.update.unit_id_to_pos(rule_inputs.params.player_idx, uid)
        )
        territory = rule_inputs.update.territory_poly()
        _, nearest_border = nearest_points(pos_point, territory.exterior)
        if (
            pos_point.distance(territory.exterior) < 10
            and not (
                near(nearest_border.x, 0)
                or near(nearest_border.x, rule_inputs.params.max_dim)
                or near(nearest_border.y, 0)
                or near(nearest_border.y, rule_inputs.params.max_dim)
            )
            and len(unit_ownership[rule_inputs.params.player_idx][uid])
            < rule_inputs.strategy_params.scout_ownership
        ):
            did_something = True
            scout_role.deallocate_unit(uid)
            defender_role.allocate_unit(uid)
    return did_something


@reallocation_rules.rule
def form_interceptors_on_threat(rule_inputs: RuleInputs):
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

    defend_zone = rule_inputs.update.territory_hull()
    for enemy in rule_inputs.update.enemies():
        for cluster_units in per_enemy_clusters[enemy].values():
            # Skip clusters that have units already targeted by Interceptors
            if any(
                cluster_unit in interceptor.target_units
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
            if defend_zone.contains(nearest_target):
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
                for _ in range(rule_inputs.strategy_params.interceptor_strength):
                    if (
                        len(defender_role.units)
                        <= rule_inputs.strategy_params.min_defenders
                    ):
                        break
                    converted_defender = defender_role.deallocation_candidate(
                        rule_inputs.update, point_to_floats(nearest_target)
                    )
                    defender_role.deallocate_unit(converted_defender)
                    interceptor_role.allocate_unit(converted_defender)
                return True
    return False


@reallocation_rules.rule
def release_interceptors_on_retreat(rule_inputs: RuleInputs):
    defender_role = rule_inputs.role_groups.of_type(RoleType.DEFENDER)[0]

    for interceptor_role in rule_inputs.role_groups.of_type(RoleType.INTERCEPTOR):
        target_positions = [
            Point(
                rule_inputs.update.unit_id_to_pos(
                    interceptor_role.target_player, target
                )
            )
            for target in interceptor_role.target_units
        ]
        if all(
            not rule_inputs.update.territory_hull().contains(target_pos)
            for target_pos in target_positions
        ):
            for uid in interceptor_role.units:
                defender_role.allocate_unit(uid)
            rule_inputs.role_groups.remove_group(
                RoleType.INTERCEPTOR, interceptor_role.id
            )
            rule_inputs.debug("Removing interceptor: targets retreated!")


@reallocation_rules.rule
def disband_interceptors_target_gone(rule_inputs: RuleInputs):
    for interceptor_role in rule_inputs.role_groups.of_type(RoleType.INTERCEPTOR):
        if len(interceptor_role.target_units) == 0:
            rule_inputs.debug("Removing interceptor: targets dead!")
            for uid in interceptor_role.units:
                rule_inputs.role_groups.of_type(RoleType.DEFENDER)[0].allocate_unit(uid)
            rule_inputs.role_groups.remove_group(
                RoleType.INTERCEPTOR, interceptor_role.id
            )
    return False


@reallocation_rules.rule
def reinforce_interceptors(rule_inputs: RuleInputs):
    defender_role = rule_inputs.role_groups.of_type(RoleType.DEFENDER)[0]

    # Sort Interceptor groups by target tile ownership
    unit_ownership, _ = rule_inputs.update.unit_ownership()
    interceptors_weighted = []
    total_interceptor_units = 0
    for interceptor_role in rule_inputs.role_groups.of_type(RoleType.INTERCEPTOR):
        total_interceptor_units += len(interceptor_role.units)
        total_target_ownership = sum(
            len(unit_ownership[interceptor_role.target_player][target_unit])
            for target_unit in interceptor_role.target_units
        )
        interceptors_weighted.append((interceptor_role, total_target_ownership))

    prioritized_interceptors = [
        interceptor_role
        for interceptor_role, _ in sorted(
            interceptors_weighted, key=lambda t: t[1], reverse=True
        )
    ]

    did_reallocations = False
    for interceptor_role in prioritized_interceptors:
        target_positions = [
            Point(
                rule_inputs.update.unit_id_to_pos(
                    interceptor_role.target_player, target
                )
            )
            for target in interceptor_role.target_units
        ]
        nearest_target_position, _ = nearest_points(
            MultiPoint(target_positions), Point(rule_inputs.params.home_base)
        )
        nearest_target_position = point_to_floats(nearest_target_position)
        current_interceptors = len(interceptor_role.units)
        if current_interceptors < rule_inputs.strategy_params.interceptor_strength:
            for _ in range(
                rule_inputs.strategy_params.interceptor_strength - current_interceptors
            ):
                if (
                    len(defender_role.units)
                    <= rule_inputs.strategy_params.min_defenders
                ):
                    return did_reallocations

                converted_defender = defender_role.deallocation_candidate(
                    rule_inputs.update, nearest_target_position
                )
                defender_role.deallocate_unit(converted_defender)
                interceptor_role.allocate_unit(converted_defender)
                did_reallocations = True
    return did_reallocations


@reallocation_rules.rule
def deploy_excess_defenders(rule_inputs: RuleInputs):
    defender_role = rule_inputs.role_groups.of_type(RoleType.DEFENDER)[0]
    target_density = (
        rule_inputs.update.territory_hull().area
        / rule_inputs.strategy_params.defender_area
    )
    did_reallocations = False
    excess_defenders = len(defender_role.units) - (1.5 * target_density)
    if excess_defenders > 0:
        if rule_inputs.params.spawn_days <= 5:
            scout_role = rule_inputs.role_groups.of_type(RoleType.SCOUT)[0]
            attacker_roles = rule_inputs.role_groups.of_type(RoleType.ATTACKER)
            for _ in range(int(excess_defenders)):
                converted_defender = defender_role.deallocation_candidate(
                    rule_inputs.update, rule_inputs.params.spawn_point
                )
                defender_role.deallocate_unit(converted_defender)

                total_scouts = len(scout_role.units)
                total_attackers = sum(
                    len(attackers.units) for attackers in attacker_roles
                )
                if total_scouts >= total_attackers:
                    global attack_roll
                    attacker_roles[attack_roll % 3].allocate_unit(converted_defender)
                    attack_roll += 1
                else:
                    scout_role.allocate_unit(converted_defender)
                did_reallocations = True
        else:
            scout_role = rule_inputs.role_groups.of_type(RoleType.SCOUT)[0]
            for _ in range(int(excess_defenders)):
                converted_defender = defender_role.deallocation_candidate(
                    rule_inputs.update, rule_inputs.params.spawn_point
                )
                defender_role.deallocate_unit(converted_defender)
                scout_role.allocate_unit(converted_defender)
                did_reallocations = True

    return did_reallocations


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
        self.params.spawn_point = point_to_floats(spawn_point)
        self.params.min_dim = min_dim
        self.params.max_dim = max_dim
        self.params.home_base = np.array(
            [(-1, -1), (-1, 101), (101, 101), (101, -1)][player_idx]
        )

        self.strategy_params = StrategyParameters(self.params)

        self.role_groups = RoleGroups()
        self.role_groups.add_group(
            RoleType.DEFENDER, LatticeDefender(self.logger, self.params)
        )

        num_attack_group = 3
        for _ in range(num_attack_group):
            self.role_groups.add_group(
                RoleType.ATTACKER, Attacker(self.logger, self.params)
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

        # Display Group Composition
        for role in RoleType:
            role_groups = self.role_groups.of_type(role)
            total_role_units = sum(len(role_group.units) for role_group in role_groups)
            self.debug(f"{role}: {total_role_units} in {len(role_groups)} groups")
        self.debug("----------")

        # Phase 1: inform roles of turn update, remove dead units
        # -------------------------------------------------------
        for role in RoleType:
            for role_group in self.role_groups.of_type(role):
                role_group.turn_update(update)

        # Phase 2: apply spawn rules
        # --------------------------
        rule_inputs = RuleInputs(
            self.logger, self.params, self.strategy_params, update, self.role_groups
        )
        for uid in just_spawned:
            spawn_rules.apply_rules(rule_inputs, uid)

        # Phase 3: apply reallocation rules
        # ---------------------------------
        reallocation_rules.apply_rules(rule_inputs)

        # Phase 4: gather moves from roles
        # --------------------------------
        moves: list[tuple[float, float]] = []
        role_moves = {}
        for role in RoleType:
            try:
                if role == RoleType.ATTACKER:
                    heatmap = density_heatmap(update.all_enemy_units())
                    # calculate heuristic
                    start_point = self.params.home_base
                    targets, avoids = target_rank(
                        update, start_point, self.player_idx, 3
                    )
                    # for each target, find a attack team best suitable for the task
                    assigned_target = assign_target(
                        update,
                        targets,
                        avoids,
                        self.role_groups.of_type(role),
                        self.params.home_base,
                        self.player_idx,
                    )
                    for role_group, target, avoid in assigned_target:
                        if len(role_group.units) > 0:
                            # for each target, also calculate its density based on heatmap
                            avoid_influence = get_avoid_influence(heatmap, target)
                            role_moves.update(
                                role_group.turn_moves(
                                    update,
                                    target=target,
                                    avoid=avoid,
                                    avoid_influence=avoid_influence,
                                )
                            )
                else:
                    for role_group in self.role_groups.of_type(role):
                        if len(role_group.units) > 0:
                            role_moves.update(role_group.turn_moves(update))
            except Exception as e:
                self.debug(
                    "Exception processing role moves:",
                    traceback.format_exc(),
                )
        for unit_id in unit_id[self.params.player_idx]:
            if not unit_id in role_moves:
                moves.append((0, 0))
            else:
                moves.append(role_moves[unit_id])

        self.turn += 1
        return moves
