import logging
import multiprocessing
import pdb
from abc import ABC, abstractmethod
from enum import Enum
from glob import glob
from os import makedirs, remove
from random import randint, uniform
from typing import Callable, Optional, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.collections import LineCollection
from scipy.spatial import KDTree
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import nearest_points, voronoi_diagram

from constants import dispute_color, player_color, tile_color

# ==================
# Game State Classes
# ==================


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
    unit_pos: list[list[Point]]
    map_states: list[list[int]]
    turn: int
    cached_ownership = Optional[
        tuple[
            dict[int, dict[str, list[tuple[int, int]]]],
            dict[tuple[int, int], tuple[int, str]],
        ]
    ]

    def __init__(
        self,
        params: GameParameters,
        turn: int,
        unit_id: list[list[str]],
        unit_pos: list[list[Point]],
        map_states: list[list[int]],
    ):
        self.params = params
        self.turn = turn
        self.unit_id = unit_id
        self.unit_pos = unit_pos
        self.map_states = map_states

    # =============================
    # Role Update Utility Functions
    # =============================

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
            for enemy_id in range(4)
            if enemy_id != self.params.player_idx
        }

    def all_enemy_units(self):
        """Returns all enemy units in a list `[(unit_id, unit_pos)]`."""
        return [
            unit for enemy_units in self.enemy_units().values() for unit in enemy_units
        ]

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
    return v / np.linalg.norm(v)


def repelling_force(p1, p2) -> tuple[float, float]:
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


class Role(ABC):
    _logger: logging.Logger
    _params: GameParameters
    _allocated_units: list[str]

    def __init__(self, logger, params):
        self._logger = logger
        self._params = params
        self.__allocated_units = []

    def _debug(self, *args):
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
    ) -> dict[str, tuple[float, float]]:
        """Returns the moves this turn for the units allocated to this role."""
        pass

    @abstractmethod
    def deallocation_candidate(self, target_point: tuple[float, float]) -> str:
        """Returns a suitable allocated unit to be de-allocated and used for other roles."""
        pass


class LatticeDefender(Role):
    __spawn_jitter: dict[str, tuple[float, float]]
    __radius: float
    counter: int

    def __init__(self, logger, params, radius):
        super().__init__(logger, params)
        self.__radius = radius
        self.__spawn_jitter = {}
        self.counter = 0

    def inside_radius(self, unit_pos: Point):
        dist = np.linalg.norm(unit_pos - self.params.home_base)
        if dist > self.__radius:
            in_vec, _ = force_vec(self.params.home_base, unit_pos)
            return in_vec
        return np.array([0, 0])

    def _turn_moves(self, update, dead_units):
        self.counter += 1
        envelope = box(0, 0, self.params.max_dim, self.params.max_dim)
        points = MultiPoint(
            [Point(pos) for pos in update.unit_pos[self.params.player_idx]]
        )
        voronoi_polys = list(voronoi_diagram(points, envelope=envelope))

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
            # Add decaying random direction bias
            if not uid in self.__spawn_jitter:
                # self.__spawn_jitter[uid] = np.array((uniform(0.1, 1), uniform(0, 0.9)))
                self.__spawn_jitter[uid] = np.array([0.0, 0.0])

            jitter_force = self.__spawn_jitter[uid]
            self.__spawn_jitter[uid] *= 0.5

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

    def deallocation_candidate(self, target_point):
        pass


class RadialDefender(Role):
    __radius: float
    # TODO: Currently assuming that newly added units are coming from spawn
    # But they might be re-allocations from other roles
    __spawn_jitter: dict[str, tuple[float, float]]

    def __init__(self, logger, params, radius):
        super().__init__(logger, params)
        self.__radius = radius
        self.__spawn_jitter = {}

    @property
    def radius(self):
        return self.__radius

    def towards_radius(self, unit_pos: Point):
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

    def deallocation_candidate(self, target_point):
        pass


class Scout(Role):
    def _turn_moves(self, update, dead_units):
        HOME_INFLUENCE = 30
        own_units = update.own_units()
        enemy_units = update.all_enemy_units()
        moves = {}
        for unit_id in self.units:
            unit_pos = own_units[unit_id]
            home_force = repelling_force(unit_pos, self.params.home_base)

            ux, uy = unit_pos
            if self.params.player_idx == 0:
                wall_normals = [(ux, self.params.min_dim), (self.params.min_dim, uy)]
            elif self.params.player_idx == 1:
                wall_normals = [(ux, self.params.max_dim), (self.params.min_dim, uy)]
            elif self.params.player_idx == 2:
                wall_normals = [(ux, self.params.max_dim), (self.params.max_dim, uy)]
            elif self.params.player_idx == 3:
                wall_normals = [(ux, self.params.min_dim), (self.params.max_dim, uy)]
            else:
                pass

            horizontal_influence = np.random.randint(40) - 10
            if int(unit_id) % 2 == 0:
                horizontal_force = repelling_force(unit_pos, wall_normals[0])
            else:
                horizontal_force = repelling_force(unit_pos, wall_normals[1])

            if int(unit_id) % 4 == 0 or int(unit_id) % 4 == 1:
                horizontal_influence = np.random.randint(30) - 10

            total_force = normalize(
                (home_force * HOME_INFLUENCE)
                + (horizontal_force * horizontal_influence)
            )
            # self._logger.debug("force", total_force)
            moves[unit_id] = to_polar(total_force)

        return moves

    def deallocation_candidate(self, target_point):
        pass


class Attacker(Role):
    def __init__(self, logger, params):
        super().__init__(logger, params)
        self.noise = dict()

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
        ATTACK_INFLUENCE = 100
        AVOID_INFLUENCE = 300

        moves = {}
        own_units = update.own_units()
        enemy_units = update.all_enemy_units()

        # TODO get attack_force
        homebase_mode = True
        # get where to initiate the attack
        units_array = np.stack([v for k, v in own_units.items()])
        if homebase_mode:  # attack from homebase
            start_point = self.params.spawn_point
        else:
            start_point = self.get_centroid(units_array)
        # get attack target and get formation
        avoid, target = self.find_target_simple(start_point, enemy_units)
        # pdb.set_trace()
        formation = LineString([start_point, target])
        # pdb.set_trace()
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

    def deallocation_candidate(self, target_point):
        pass


# =======================================
# Dynamic Allocation Rules Infrastructure
# =======================================
class RuleInputs:
    logger: logging.Logger
    params: GameParameters
    update: StateUpdate
    role_groups: dict[RoleType, list[Role]]

    def __init__(
        self,
        logger: logging.Logger,
        params: GameParameters,
        update: StateUpdate,
        role_groups: dict[RoleType, list[Role]],
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
def populate_defenders(rule_inputs: RuleInputs, uid: str):
    if (
        rule_inputs.update.turn > 30
        and len(rule_inputs.role_groups[RoleType.DEFENDER][0].units) < 20
    ):
        rule_inputs.role_groups[RoleType.DEFENDER][0].allocate_unit(uid)
        return True
    return False


@spawn_rules.rule
def even_scouts_attackers(rule_inputs: RuleInputs, uid):
    total_scouts = sum(
        len(scouts.units) for scouts in rule_inputs.role_groups[RoleType.SCOUT]
    )
    total_attackers = sum(
        len(attackers.units) for attackers in rule_inputs.role_groups[RoleType.ATTACKER]
    )
    if total_scouts >= total_attackers:
        rule_inputs.role_groups[RoleType.ATTACKER][0].allocate_unit(uid)
    else:
        rule_inputs.role_groups[RoleType.SCOUT][0].allocate_unit(uid)
    rule_inputs.debug("scout/attacker rule")
    return True


# =======================
# Role Reallocation Rules
# =======================
# Order of rules declared below is their precedence
# Higher up means higher precedence
reallocation_rules = ReallocationRules()


@reallocation_rules.rule
def remove_empty_attack_groups(rule_inputs: RuleInputs):
    if any(
        len(group.units) == 0 for group in rule_inputs.role_groups[RoleType.ATTACKER]
    ):
        rule_inputs.role_groups[RoleType.ATTACKER] = [
            group
            for group in rule_inputs.role_groups[RoleType.ATTACKER]
            if len(group.units) > 0
        ]
        return True
    return False


# ======
# Player
# ======
class Player:
    params: GameParameters
    logger: logging.Logger
    turn: int
    known_units: set[str]

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

        self.role_groups: dict[RoleType, list[Role]] = {role: [] for role in RoleType}
        self.role_groups[RoleType.DEFENDER].append(
            LatticeDefender(self.logger, self.params, 40)
        )
        self.role_groups[RoleType.ATTACKER].append(Attacker(self.logger, self.params))
        self.role_groups[RoleType.SCOUT].append(Scout(self.logger, self.params))

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
                [min(100, (750 / (d1) + 750 / (d2))) for d1, d2 in risk_distances],
            )
        )
        visualize_risk(risks, enemy_unit_locations, own_units, self.turn)

        # Calculate free units (just spawned)
        own_units = set(uid for uid in unit_id[self.params.player_idx])
        allocated_units = set(
            uid
            for role_group in self.role_groups.values()
            for role in role_group
            for uid in role.units
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
            for role_group in self.role_groups[role]:
                role_moves.update(role_group.turn_moves(update))
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
