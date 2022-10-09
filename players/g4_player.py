import os
import pickle
import numpy as np
import sympy
import logging
from typing import Tuple
import matplotlib.pyplot as plt
from constants import player_color, tile_color, dispute_color, base
from matplotlib import colors
import pandas as pd

def sympy_p_float(p: sympy.Point2D):
    return np.array([float(p.x), float(p.y)])


class Player:
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
        self.turn = 0

        # Game fundamentals
        self.total_days = total_days
        self.spawn_days = spawn_days
        self.player_idx = player_idx
        self.spawn_point = spawn_point
        self.min_dim = min_dim
        self.max_dim = max_dim

        if self.player_idx == 0:
            self.homebase = (-1, -1)
        elif self.player_idx == 1:
            self.homebase = (-1, 101)
        elif self.player_idx == 2:
            self.homebase = (101, 101)
        else:
            self.homebase = (101, -1)

    def debug(self, *args):
        self.logger.info(" ".join(str(a) for a in args))

    def clamp(self, x, y):
        return (
            min(self.max_dim, max(self.min_dim, x)),
            min(self.max_dim, max(self.min_dim, y)),
        )

    def force_vec(self, p1, p2):
        v = p1 - p2
        mag = np.linalg.norm(v)
        unit = v / mag
        return unit, mag

    def to_polar(self, p):
        x, y = p
        return np.sqrt(x**2 + y**2), np.arctan2(y, x)

    def normalize(self, v):
        return v / np.linalg.norm(v)

    def repelling_force(self, p1, p2):
        dir, mag = self.force_vec(p1, p2)
        # Inverse magnitude: closer things apply greater force
        return dir * 1 / (mag)


    def risk_distances(self, enemy_location, own_units):
       # self.debug("\thomebase:", self.homebase)
        #self.debug("\tEnemy location:",enemy_location)
        #self.debug("\tEnemy distance_d_home_base:",np.linalg.norm(np.subtract(self.homebase,enemy_location)))
        d_base = np.linalg.norm(np.subtract(self.homebase,enemy_location))
        d_to_closest_unit = 150
        for unit in own_units:
            d_our_unit = np.linalg.norm(np.subtract(unit[1], enemy_location))
            d_to_closest_unit = min(d_our_unit,d_to_closest_unit)
        return (d_base, d_to_closest_unit)

    def play(
        self, unit_id, unit_pos, map_states, current_scores, total_scores
    ) -> [tuple[float, float]]:
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

        # (id, (x, y))
        DISPLAY_EVERY_N_ROUNDS = 30
        HEAT_MAP = True
        
        own_units = list(
            zip(
                unit_id[self.player_idx],
                [sympy_p_float(pos) for pos in unit_pos[self.player_idx]],
            )
        )
        enemy_units_locations = [
            sympy_p_float(unit_pos[player][i])
            for player in range(len(unit_pos))
            for i in range(len(unit_pos[player]))
            if player != self.player_idx
        ]

        #self.logger.info(own_units)
        #self.logger.info(enemy_units_locations)
       # self.logger.info(own_units)
       # self.logger.info(enemy_units_locations)
        #self.debug("\tEnemy unit locations:", enemy_units_locations)
        risk_distances = [
            self.risk_distances(enemy_location, own_units)
            for enemy_location in enemy_units_locations
        ]

        self.debug("\tRisk distances:", risk_distances)

        risks = list(
            zip(enemy_units_locations,
                [min(100,(750/(d1)+750/(d2))) for d1, d2 in risk_distances]
            )
        )
        
        self.debug("\tRisks:", risks)

        ENEMY_INFLUENCE = 1
        HOME_INFLUENCE = 20
        ALLY_INFLUENCE = 0.5

        moves = []
        for unit_id, unit_pos in own_units:
            self.debug(f"Unit {unit_id}", unit_pos)
            enemy_unit_forces = [
                self.repelling_force(unit_pos, enemy_pos)
                for enemy_pos in enemy_units_locations
            ]
            enemy_force = np.add.reduce(enemy_unit_forces)

            ally_forces = [
                self.repelling_force(unit_pos, ally_pos)
                for ally_id, ally_pos in own_units
                if ally_id != unit_id
            ]
            ally_force = np.add.reduce(ally_forces)

            home_force = self.repelling_force(unit_pos, self.homebase)
            self.debug("\tEnemy force:", enemy_force)
            self.debug("\tHome force:", home_force)

            total_force = self.normalize(
                (enemy_force * ENEMY_INFLUENCE)
                + (home_force * HOME_INFLUENCE)
                + (ally_force * ALLY_INFLUENCE)
            )
            self.debug("\tTotal force:", total_force)

            moves.append(self.to_polar(total_force))

        if HEAT_MAP and self.turn % DISPLAY_EVERY_N_ROUNDS == 0:

            c = []
            for r in risks:
                c.append(r[1])
            plt.rcParams["figure.autolayout"] = True
            x = np.array(enemy_units_locations)[:,0]
            y = np.array(enemy_units_locations)[:,1]
            c = np.array(c)
       
            self.logger.info(c)
            df = pd.DataFrame({"x": x, "y": y, "c": c})
                
            fig, ax = plt.subplots()#1,1, figsize=(20,6))
            cmap = plt.cm.hot
            norm = colors.Normalize(vmin=0.0, vmax=100.0)
            mp = ax.scatter(df.x, df.y, color=cmap(norm(df.c.values)))
            ax.set_xticks(df.x)

            fig.subplots_adjust(right=0.9)
            sub_ax = plt.axes([.8,.4,.1,.4])
            

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            plt.colorbar(sm,cax=sub_ax)
            ax.invert_yaxis()

            for p in range(1):
                for num, pos in own_units:
                    ax.scatter(
                        pos[0],
                        pos[1],
                        color="blue"
                    )

            np.meshgrid(list(range(100)), list(range(100)))
            plt.title(f"Day {self.turn}")
            plt.show()
        self.turn += 1
        return moves
