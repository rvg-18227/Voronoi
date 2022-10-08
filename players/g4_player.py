import os
import pickle
from re import L
from xml.dom.minidom import parseString
import numpy as np
import sympy
import logging
import pdb
from typing import Tuple
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

EPSILON = 0.0000001

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
        
    
    def attack_point(self, units, target, homebase_mode=True):
        '''
        Given a list of unit, attack the target point in a line formation.
        Return: A list of attack move using units for the target.
        '''
        # Intuition:
        #    We want to form an line first before attacking
        #    But it is not always the optimal move
        #    Since it takes time to form a line, during which enemy might change formation
        #    Which might make our attack useless
        #    We also dont want to shove all attack unit toward the target
        #    Since that would most likely to be suicidal
        #    So we want to leverage between forming a formation and moving towards the target
        #    SCALE BY DISTANCE:
        #        Unit further away from the expected line foramtion should move closer to line
        #        Unit closer to the line should move toward the target
        #    Problem FOR NOW:
        #        How to space our our unit in a more even/tight manner
        #    Its is better to attack from homebase.
        #       Forming line that deviate from homebase is suspectiable from side attack from another players
        #       But attack from the centroid of the units is more effective
        def get_centroid(units):
            '''
            Find centroid on a cluster of points
            '''
            return units.mean(axis=0)
        
        def find_closest_point(line, point):
            '''
            Find closest point on line segment given a point
            '''
            return nearest_points(line, point)[0]
        
        def compute_vector(units, closest_point, target_point):
            # given units, their corresponding cloest point in the attack line, and a target
            # compute unit vector for attacking
            move = []
            for i in range(len(units)):
                unit_vec_closest, mag_closest = self.force_vec(units[i], closest_point[i].coords)
                unit_vec_target, mag_target = self.force_vec(units[i], target_point)
                # When magnitude to target is large, it mean it is close to target
                # So move toward the closet point to line
                total_mag = mag_target + mag_closest
                weight_target = mag_target/total_mag
                # Same as 
                weight_closest = mag_closest/total_mag
                #pdb.set_trace()
                move_vec = unit_vec_closest * weight_closest + unit_vec_target * weight_target
                move_vec = move_vec/np.linalg.norm(move_vec)
                move_vec *= -1
                move.append(move_vec)
                #pdb.set_trace()
                self.debug("unit has move vector", move_vec)
            #pdb.set_trace()
            return move
            
        if homebase_mode:# attack from homebase
            start_point = self.spawn_point
        else:
            start_point = get_centroid(units)
        line = LineString([start_point, target])
        cloest_points = []
        for i in units:
            closest_pt_to_line = find_closest_point(line, Point(i))
            cloest_points.append(closest_pt_to_line)
        move = compute_vector(units, cloest_points, target)
        move = [self.to_polar((x[0][0], x[0][1])) for x in move]
        return move
            
    def find_weak_points(self, num_weak_pts, enemy_units):
        '''
        Given an enemy player and their unit, find weak points = num_weak_pts
        '''
        # TODO implement heuristic
        # Right now we are using the two most sparse point in enemy units as "weak point"
        # Using rectangular sampling region for now
        # How to measure "weakness" given enemy formation?
        # TODO
        # How to make the sampling region "smarter"?
        # Random sampling? What about spacing?
        # NEED HEAT MAP
        # Given heat map and map state, we can find weak point.
        return [25, 75], [75, 50]
    
    def get_enemy_unit(self, enemy_idx, unit_pos):
        '''
        Given an enemy player index, return the unit location of enemy player i.
        '''
        return unit_pos[enemy_idx]
    
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
        unit = v / (mag + EPSILON)
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
        
        #if len(unit_id[0]) > 1:
        #pdb.set_trace()
        # assume we are always player 1
        # assume we are attacking player 2 with all we have (singular enemy)
        enemy2 = self.get_enemy_unit(1, unit_pos)
        weak_pt_player_2 = self.find_weak_points(2, enemy2)
        my_point = np.stack([sympy_p_float(pos) for pos in unit_pos[self.player_idx]], axis=0)
        attack_move = self.attack_point(my_point, weak_pt_player_2[0])
        return attack_move

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

        return moves
