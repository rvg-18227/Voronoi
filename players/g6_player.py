import os
import pickle
import numpy as np
import sympy
import logging
from typing import Tuple
from sympy.geometry import Point2D
from enum import Enum

class UnitType(Enum):
    SPACER = 0
    ATTACK = 1
    DEFENSE = 2

class Defender:
    def __init__(self, id, position):
        self.id = id
        self.x = float(position[0])
        self.y = float(position[1])
        self.unit_type = UnitType.DEFENSE
    def get_move(self, game):
        return self.x, self.y, 1



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
        self.player_idx = player_idx
        self.total_days = total_days
        self.spawn_days = spawn_days
        self.spawn_point = spawn_point

        self.current_turn = 0
        self.PHASE_ONE_OUTPUT = [UnitType.SPACER]
        self.PHASE_TWO_OUTPUT = [UnitType.SPACER, UnitType.ATTACK, UnitType.DEFENSE, UnitType.ATTACK, UnitType.DEFENSE]
        self.PHASE_THREE_OUTPUT = [UnitType.ATTACK, UnitType.DEFENSE, UnitType.ATTACK, UnitType.DEFENSE]

        self.number_units_total = total_days//spawn_days
        self.PHASE_ONE_UNITS = 5
        self.PHASE_TWO_UNITS = min(len(self.PHASE_TWO_OUTPUT), np.floor((self.number_units_total-5)*0.6))
        self.PHASE_THREE_UNITS = self.number_units_total - self.PHASE_ONE_UNITS - self.PHASE_TWO_UNITS

        #locations of units
        self.unit_types = {
            UnitType.SPACER: {},
            UnitType.ATTACK: {},
            UnitType.DEFENSE: {}
        }


    def play(self, unit_id, unit_pos, map_states, current_scores, total_scores) -> [tuple[float, float]]:
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
        self.current_turn += 1
        self.add_spawn_units_if_needed(unit_id[self.player_idx])
        spacer, attacker, defenders = self.get_unit_indexes(unit_id[self.player_idx]) 

        # 3 roles
        # attackers - Identify weak enemy, how to kill a unit? where to attack? when to attack? hover at border until enough units? whats the best formation?
        # defenders - kmeans number of units around/units in a radius, most important!!!, so be able to use other groups if needed
        # space gain people - look at group 4's code or come up wiht a new generic strategy

        # 3 phases
        # in intro phase allocate X spacegain until 5 units?
        # in main phase allocate X attackers and Y defenders and Z spacegain 2:2:1, maybe change ratio based on if we're being attacked or not
        # in end phase allocate X attackers and Y defenders in some proportion 

        self.map_states = map_states
        moves = [self.transform_move(0, 0, 0)] * len(unit_pos[self.player_idx])

        for idx in spacer:
            moves[idx] = self.transform_move(1, 1) #spacer.move_function(unit, idx, etc)

        for idx in attacker:
            moves[idx] = self.transform_move(0, 1) #attacker.move_function(unit, idx, etc)

        for idx in defenders:
            defender = Defender(unit_id[self.player_idx][idx], unit_pos[self.player_idx][idx])
            x, y, dist = defender.get_move(self.map_states)
            moves[idx] = self.transform_move(x, y, dist)

        return moves

    def simulate_move(self, unit_pos, move) -> tuple[float, float]:
        """Simulates the move and returns the new position
                Args:
                    unit_pos (Point2D(float, float)): current position of the unit
                    move (tuple(float, float)): move to be made as (distance, angle)
                Returns
                    tuple(float, float): new position of the unit
        """
        distance, angle = move
        return (unit_pos[0] + distance * np.cos(angle), unit_pos[1] + distance * np.sin(angle))

    def check_square(self, pos):
        """Checks who owns the square of a given position
                Args:
                    pos (Point2D(float, float)): current position
                Returns:
                    int: player index of the player who owns the cell
        """
        return self.map_states[np.floor(pos[0])][np.floor(pos[1])] - 1 # -1 because the map states are 1 indexed

    def transform_move(self, x: float, y: float, distance=1) -> tuple[float, float]:
        """Transforms the distance and angle to the correct format for the game engine
                Args:
                    angle (float): angle in radians
                    distance (float): distance to move
                Returns
                    Tuple[float, float]: distance and angle in correct format
        """
        angle = np.arctan2(y, x)
        if self.player_idx == 0:
            return (distance, angle)
        elif self.player_idx == 1:
            return (distance, angle - np.pi/2)
        elif self.player_idx == 2:
            return (distance, angle + np.pi)
        else:# self.player_idx == 3:
            return (distance, angle + np.pi / 2)

    def add_spawn_units_if_needed(self, unit_ids):
        if self.current_turn % self.spawn_days == 0:
            if self.current_turn <= self.PHASE_ONE_UNITS * self.spawn_days:
                unitToAdd = self.PHASE_ONE_OUTPUT[(self.current_turn//self.number_units_total )%len(self.PHASE_ONE_OUTPUT)]
            elif self.current_turn <= (self.PHASE_ONE_UNITS + self.PHASE_TWO_UNITS) * self.spawn_days:
                numberDaysThisPhase = self.current_turn - self.PHASE_ONE_UNITS * self.spawn_days - 1
                unitToAdd = self.PHASE_TWO_OUTPUT[(numberDaysThisPhase//self.spawn_days)%len(self.PHASE_TWO_OUTPUT)]
            else:
                numberDaysThisPhase = int(self.current_turn - (self.PHASE_ONE_UNITS + self.PHASE_TWO_UNITS) * self.spawn_days - 1)
                unitToAdd = self.PHASE_THREE_OUTPUT[(numberDaysThisPhase//self.spawn_days)%len(self.PHASE_THREE_OUTPUT)]
            self.unit_types[unitToAdd][unit_ids[len(unit_ids)-1]] = len(unit_ids)-1

    def get_unit_indexes(self, unit_ids):
        """Returns the indexes of the units in the unit_pos list by type
                Args:
                    unit_ids (list(str)): list of unit ids
                Returns:
                    Tuple[list(Point2D), list(Point2D), list(Point2D))]: indexes of the units in the unit_pos list by type
        """
        spacer = [idx for id, idx in self.unit_types[UnitType.SPACER].items() if id in unit_ids]
        attacker = [idx for id, idx in self.unit_types[UnitType.ATTACK].items() if id in unit_ids]
        defender = [idx for id, idx in self.unit_types[UnitType.DEFENSE].items() if id in unit_ids]

        return spacer, attacker, defender
        