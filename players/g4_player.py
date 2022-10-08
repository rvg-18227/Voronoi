import numpy as np
from shapely.geometry import Point
import logging
from typing import Tuple
from abc import ABC, abstractmethod
from enum import Enum
import pdb


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

    unit_id: list[list[str]]
    unit_pos: list[list[Point]]


# =======================
# Force Utility Functions
# =======================
def force_vec(p1, p2):
    """Vector direction and magnitude pointing from `p2` to `p1`"""
    v = p1 - p2
    mag = np.linalg.norm(v)
    unit = v / mag
    return unit, mag


def to_polar(p):
    x, y = p
    return np.sqrt(x**2 + y**2), np.arctan2(y, x)


def normalize(v):
    return v / np.linalg.norm(v)


def repelling_force(p1, p2) -> tuple[float, float]:
    dir, mag = force_vec(p1, p2)
    # Inverse magnitude: closer things apply greater force
    return dir * 1 / (mag)


def linear_attracting_force(p1, p2):
    return force_vec(p2, p1)


class RoleType(Enum):
    DEFENDER = 1
    ATTACKER = 2


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
        alive_units = set(self._own_units(update).keys())
        allocated_set = set(self.__allocated_units)
        dead_units = allocated_set - alive_units
        self.__allocated_units = list(allocated_set - dead_units)

        return self._turn_moves(update, dead_units)

    # =============================
    # Role Update Utility Functions
    # =============================

    def _own_units(self, update: StateUpdate):
        return {
            unit_id: unit_pos
            for unit_id, unit_pos in zip(
                update.unit_id[self.params.player_idx],
                [pos for pos in update.unit_pos[self.params.player_idx]],
            )
        }

    def _enemy_units(self, update: StateUpdate):
        return {
            enemy_id: list(zip(update.unit_id[enemy_id], update.unit_pos[enemy_id]))
            for enemy_id in range(4)
            if enemy_id != self.params.player_idx
        }

    def _all_enemy_units(self, update: StateUpdate):
        return [
            unit
            for enemy_units in self._enemy_units(update).values()
            for unit in enemy_units
        ]

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


class Defender(Role):
    def _turn_moves(self, update, dead_units):
        ENEMY_INFLUENCE = 1
        HOME_INFLUENCE = 20
        ALLY_INFLUENCE = 0.5
        WALL_INFLUENCE = 1

        moves = {}
        own_units = self._own_units(update)
        enemy_units = self._all_enemy_units(update)

        for unit_id in self.units:
            unit_pos = own_units[unit_id]

            enemy_unit_forces = [
                repelling_force(unit_pos, enemy_pos) for _, enemy_pos in enemy_units
            ]
            enemy_force = np.add.reduce(enemy_unit_forces)

            ally_forces = [
                repelling_force(unit_pos, ally_pos)
                for ally_id, ally_pos in own_units.items()
                if ally_id != unit_id
            ]
            ally_force = np.add.reduce(ally_forces)

            home_force = repelling_force(unit_pos, self.params.home_base)

            ux, uy = unit_pos
            wall_normals = [
                (ux, self.params.min_dim),
                (ux, self.params.max_dim),
                (self.params.min_dim, uy),
                (self.params.max_dim, uy),
            ]
            wall_forces = [repelling_force(unit_pos, wall) for wall in wall_normals]
            wall_force = np.add.reduce(wall_forces)

            total_force = normalize(
                (enemy_force * ENEMY_INFLUENCE)
                + (home_force * HOME_INFLUENCE)
                + (ally_force * ALLY_INFLUENCE)
                + (wall_force * WALL_INFLUENCE)
            )

            moves[unit_id] = to_polar(total_force)

        return moves

    def deallocation_candidate(self, target_point):
        pass


class NaiveAttacker(Role):
    def _turn_moves(self, update, dead_units):
        ENEMY_INFLUENCE = -1
        HOME_INFLUENCE = 20
        ALLY_INFLUENCE = 0.5
        WALL_INFLUENCE = 1

        moves = {}
        own_units = self._own_units(update)
        enemy_units = self._all_enemy_units(update)

        for unit_id in self.units:
            unit_pos = own_units[unit_id]

            enemy_unit_forces = [
                repelling_force(unit_pos, enemy_pos) for _, enemy_pos in enemy_units
            ]
            enemy_force = np.add.reduce(enemy_unit_forces)

            ally_forces = [
                repelling_force(unit_pos, ally_pos)
                for ally_id, ally_pos in own_units.items()
                if ally_id != unit_id
            ]
            ally_force = np.add.reduce(ally_forces)

            home_force = repelling_force(unit_pos, self.params.home_base)

            ux, uy = unit_pos
            wall_normals = [
                (ux, self.params.min_dim),
                (ux, self.params.max_dim),
                (self.params.min_dim, uy),
                (self.params.max_dim, uy),
            ]
            wall_forces = [repelling_force(unit_pos, wall) for wall in wall_normals]
            wall_force = np.add.reduce(wall_forces)

            total_force = normalize(
                (enemy_force * ENEMY_INFLUENCE)
                + (home_force * HOME_INFLUENCE)
                + (ally_force * ALLY_INFLUENCE)
                + (wall_force * WALL_INFLUENCE)
            )

            moves[unit_id] = to_polar(total_force)

        return moves

    def deallocation_candidate(self, target_point):
        pass


class Player:
    params: GameParameters

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
        self.role_groups[RoleType.DEFENDER].append(Defender(self.logger, self.params))
        self.role_groups[RoleType.ATTACKER].append(
            NaiveAttacker(self.logger, self.params)
        )

    def debug(self, *args):
        self.logger.info(" ".join(str(a) for a in args))

    def clamp(self, x, y):
        return (
            min(self.max_dim, max(self.min_dim, x)),
            min(self.max_dim, max(self.min_dim, y)),
        )

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

        ENEMY_INFLUENCE = 1
        HOME_INFLUENCE = 20
        ALLY_INFLUENCE = 0.5
        WALL_INFLUENCE = 1

        # Calculate free units (just spawned)
        own_units = set(uid for uid in unit_id[self.params.player_idx])
        allocated_units = set(
            uid
            for role_group in self.role_groups.values()
            for role in role_group
            for uid in role.units
        )
        free_units = own_units - allocated_units

        # Naive split allocation between attackers and defenders
        for uid in free_units:
            total_defenders = sum(
                len(defenders.units)
                for defenders in self.role_groups[RoleType.DEFENDER]
            )
            total_attackers = sum(
                len(attackers.units)
                for attackers in self.role_groups[RoleType.ATTACKER]
            )
            if total_defenders >= total_attackers:
                self.role_groups[RoleType.ATTACKER][0].allocate_unit(uid)
            else:
                self.role_groups[RoleType.DEFENDER][0].allocate_unit(uid)

        # Convert unit positions to floats
        unit_pos = [
            [point_to_floats(player_unit_pos) for player_unit_pos in player_units]
            for player_units in unit_pos
        ]

        update = StateUpdate()
        update.unit_id = unit_id
        update.unit_pos = unit_pos

        moves: list[tuple[float, float]] = []
        role_moves = {}
        for role in RoleType:
            for role_group in self.role_groups[role]:
                role_moves.update(role_group.turn_moves(update))
        for unit_id in unit_id[self.params.player_idx]:
            moves.append(role_moves[unit_id])
        return moves
