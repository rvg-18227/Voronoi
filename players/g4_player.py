import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
import logging
import pdb
from typing import Tuple
from abc import ABC, abstractmethod
from enum import Enum
import pdb
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
from random import uniform, randint

EPSILON = 0.0000001


def point_to_floats(p: Point):
    return np.array([float(p.x), float(p.y)])


class GameParameters:
    """Represents constant game parameters that don't change."""

    total_days: int
    spawn_days: int
    player_idx: int
    spawn_point: tuple[float, float]
    min_dim: int
    max_dim: int
    home_base: tuple[float, float]


class StateUpdate:
    """Represents all of the data that changes between turns."""

    params: GameParameters
    unit_id: list[list[str]]
    unit_pos: list[list[Point]]

    def __init__(
        self,
        params: GameParameters,
        unit_id: list[list[str]],
        unit_pos: list[list[Point]],
    ):
        self.params = params
        self.unit_id = unit_id
        self.unit_pos = unit_pos

    # =============================
    # Role Update Utility Functions
    # =============================

    def own_units(self):
        return {
            unit_id: unit_pos
            for unit_id, unit_pos in zip(
                self.unit_id[self.params.player_idx],
                [pos for pos in self.unit_pos[self.params.player_idx]],
            )
        }

    def enemy_units(self):
        return {
            enemy_id: list(zip(self.unit_id[enemy_id], self.unit_pos[enemy_id]))
            for enemy_id in range(4)
            if enemy_id != self.params.player_idx
        }

    def all_enemy_units(self):
        return [
            unit for enemy_units in self.enemy_units().values() for unit in enemy_units
        ]


# =======================
# Force Utility Functions
# =======================
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


class Player:
    params: GameParameters
    logger: logging.Logger
    turn: int

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
            RadialDefender(self.logger, self.params, radius=30)
        )
        self.role_groups[RoleType.ATTACKER].append(Attacker(self.logger, self.params))
        self.role_groups[RoleType.SCOUT].append(Scout(self.logger, self.params))

    def debug(self, *args):
        self.logger.info(" ".join(str(a) for a in args))

    def clamp(self, x, y):
        return (
            min(self.max_dim, max(self.min_dim, x)),
            min(self.max_dim, max(self.min_dim, y)),
        )

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

        update = StateUpdate(self.params, unit_id, unit_pos)

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

        RING_SPACING = 5
        MIN_RADIUS = 5
        idle = 0
        for uid in free_units:
            assigned = False

            # TODO: framework for prioritizing allocation rules

            # Currently assuming all defenders are RadialDefender
            # Also assumes that defenders are ordered by priority for reinforcement
            for ring in reversed(self.role_groups[RoleType.DEFENDER]):
                target_density = int((np.pi * ring.radius / 2) / RING_SPACING)
                if len(ring.units) < target_density:
                    ring.allocate_unit(uid)
                    assigned = True

            if assigned:
                continue

            last_ring: RadialDefender = self.role_groups[RoleType.DEFENDER][-1]
            # (1/4 circle circumference) / (spacing between units) = units in ring
            target_density = int((np.pi * last_ring.radius / 2) / RING_SPACING)
            next_radius = last_ring.radius / 2
            if len(last_ring.units) >= target_density and next_radius >= MIN_RADIUS:
                self.debug(f"Creating new Defender ring with radius {next_radius}")
                last_ring = RadialDefender(self.logger, self.params, next_radius)
                self.role_groups[RoleType.DEFENDER].append(last_ring)

                last_ring.allocate_unit(uid)
                assigned = True

            if assigned:
                continue

            total_scouts = sum(
                len(scouts.units) for scouts in self.role_groups[RoleType.SCOUT]
            )
            total_attackers = sum(
                len(attackers.units)
                for attackers in self.role_groups[RoleType.ATTACKER]
            )
            if total_scouts >= total_attackers:
                self.role_groups[RoleType.ATTACKER][0].allocate_unit(uid)
                assigned = True
            else:
                self.role_groups[RoleType.SCOUT][0].allocate_unit(uid)
                assigned = True

            if assigned:
                continue

            idle += 1

        if idle > 0:
            self.debug(f"Turn {self.turn}: {idle} idle units")

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
