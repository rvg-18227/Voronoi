from ast import Num
from cmath import acos
from copy import deepcopy
from doctest import DocFileSuite
import os
import pickle
from re import L
from threading import currentThread
from turtle import distance
import numpy as np
import sympy
import logging
from typing import Tuple
from math import pi, dist
import numpy as np
from scipy.spatial.distance import cdist


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

        # Default variables passed into function
        self.rng = rng
        self.logger = logger
        self.spawn_days = spawn_days
        self.total_days = total_days
        self.spawn_point = [spawn_point.x, spawn_point.y]
        self.player_idx = player_idx
        self.min_dim = min_dim
        self.max_dim = max_dim
        if self.player_idx == 0: # Top left orange
            self.angles = [(2*pi/8), (4*pi/8), 0, (3*pi/8), (pi/8) ]
        elif self.player_idx == 1: # Bottom left pink
            self.angles = [7*pi/4, 3*pi/2, 0, 13*pi/8, 15*pi/8] 
        elif self.player_idx == 2: # Bottom right blue
            self.angles = [5*pi/4, 3*pi/2, pi, 11*pi/8, 9*pi/8] 
        else: # Top right yellow
            self.angles = [3*pi/4, pi, pi/2, 7*pi/8, 5*pi/8]

        # Experimenal variables
        self.day = 0
        self.other_players = [0,1,2,3]
        self.other_players.remove(self.player_idx)
        self.last_n_days_positions = {}
        self.hostility_registry = {}
        self.units_angle = {}

        for other_player in self.other_players:
            self.last_n_days_positions[other_player] = []
            self.hostility_registry[other_player] = 1

    # Find the nearest unit to a given space
    def nearest_unit_to_space(self, team_unit_ids, team_unit_pos, space):
        pass

    # Gives a map of troop density is lowest
    def density_map(self):
        pass


    def basic_aggressiveness(self, prev_position, current_position) -> float:
        """Function which based on current and past posisitions returns the difference in avg distance from spawn point.
            Is fairly inaccurate due to summing all units distance when one or two units could cause lots of damage in a sparse game

                Args:
                    unit_id (list(list(str))): contains the position of each unit on one team at some previous time
                    current_position (list(list(float))): contains the position of each unit on one team currently
                Returns:
                    float: Where greater values indicate more aggressive players
                """
        total_prev = 0
        total_current = 0
        for p in prev_position:
            total_prev = abs(dist(self.spawn_point, [p.x, p.y]))

        for p in current_position:
            total_current = abs(dist(self.spawn_point, [p.x, p.y]))

        return total_current/len(current_position) - total_prev/len(prev_position)
    
    # finds the player with the most score besides us
    def currentEnemyWinner(self, current_scores) -> list[int]:
        score, player_idx1 = -1, -1
        for other in self.other_players:
            if current_scores[other] > score:
                score = current_scores[other]
                player_idx1 = other
        return self.inCollaboration(player_idx1, current_scores)
        
    # determines whehter a player is in collaboration with another player
    def inCollaboration(self, firstEnemy, current_scores) -> int:
        remainingPlayer = self.other_players.copy()
        remainingPlayer.remove(firstEnemy)
        score, secondEnemy = -1, -1
        for other in remainingPlayer:
            if current_scores[other] > score:
                score = current_scores[other]
                secondEnemy = other
        if current_scores[firstEnemy] !=0 and current_scores[secondEnemy]/current_scores[firstEnemy] > 0.96:
            return [firstEnemy, secondEnemy]
        return [firstEnemy]
    
    # return the two points I should be moving toward player direction.
    def findTwoClosest(self, unit_pos, player):
        pos_to_me = [[None], [None]]
        distances = cdist(list([i.x, i.y] for i in unit_pos[self.player_idx]), list([i.x, i.y] for i in unit_pos[player]), metric='euclidean')
        
        globalFirst, globalFirstIdx = 1000000, -1
        for myPoint, otherList in enumerate(distances):
            first, firstIdx = 1000000, -1
            for idx, num in enumerate(otherList):
                if num < first:
                    first, firstIdx = num, idx
                    otherList[idx] = 100000
            if first < globalFirst:
                globalFirst, globalFirstIdx = first, firstIdx
                pos_to_me[0] = [myPoint, globalFirstIdx]

        globalSecond, globalSecondIdx = 1000000, -1
        for myPoint, otherList in enumerate(distances):
            if myPoint == pos_to_me[0][0] and myPoint != pos_to_me[1][0]:
                first, firstIdx = 1000000, -1
                for idx, num in enumerate(otherList):
                    if num < first:
                        first, firstIdx = num, idx
                        otherList[idx] = 100000
                if first < globalSecond:
                    globalSecond, globalSecondIdx = first, firstIdx
                    pos_to_me[1] = [myPoint, globalSecondIdx]
        return pos_to_me
    
    def moveTowardAggressive(self, current_scores, unit_pos, unit_id):
        # returns which of our unit should be move toward the middle of the enemy's closest two points
        agg_players = self.currentEnemyWinner(current_scores)
        strategies = []
        # used = []
        for player in agg_players:
            
            two_points_close = self.findTwoClosest(unit_pos, player)
            
            
            # there could the case where two_points_close[0][0] is different from two_points_close[1][0]
            # this would be rare, thus not handle 
            my_unit_idx = two_points_close[0][0]
            # while my_unit_idx in used:
            #     copied = unit_pos.copy()
            #     copy = copied[self.player_idx]
            #     print(copy)
            #     return 0
            #     # two_points_close = self.findTwoClosest(copy, player)
            ene_unit_1_idx, ene_unit_2_idx = two_points_close[0][1], two_points_close[1][1]
            
            a  = unit_pos[self.player_idx][my_unit_idx] # my point
            b  = unit_pos[player][ene_unit_1_idx] # b = ene point_1
            c  = unit_pos[player][ene_unit_2_idx] # c = ene point_2
            
            ab, ac, bc = dist((a.x, a.y), (b.x, b.y)), dist((a.x, a.y), (c.x, c.y)), dist((b.x, b.y), (c.x, c.y))
            mid_angle = (acos((ac**2 - ab**2 - bc**2)/(-2.0 * ab * ac))) / 2
            
            strategies.append({'move': unit_id[self.player_idx][my_unit_idx],'player': player, 'ene_points':[ene_unit_1_idx,ene_unit_2_idx],  'mid_angle': mid_angle.real})
            # used.append(my_unit_idx) # will be used to move 

            # print('player, :', player, 'mid_angle', mid_angle*180/pi, '(my_unit_id, ene1_id, ene2_id): (', unit_id[self.player_idx][my_unit_idx], unit_id[player][ene_unit_1_idx], unit_id[player][ene_unit_2_idx], ')')
        
        return strategies

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
        total_units = len(unit_id[self.player_idx])
        friendly_unit_ids = unit_id[self.player_idx]
        moves = []
        locx = 0
        locy = 0

        dead_units = [unit for unit in self.units_angle if unit not in friendly_unit_ids]
        for dead_units in dead_units:
            del self.units_angle[dead_units]

    
        offense = self.moveTowardAggressive(current_scores, unit_pos, unit_id)
        offense_idx = [i['move'] for i in offense]

        if self.player_idx == 0:
            locx = 0
            locy = 0
        elif self.player_idx == 1:
            locx = 0
            locy = 75
        elif self.player_idx == 2:
            locx = 75
            locy = 75
        elif self.player_idx == 3:
            locx = 75
            locy = 0

        attackers = []
        if self.day > 49:
            for i in range(25-1):
                for j in range(25-1):
                    if map_states[i + locx][j + locy] != self.player_idx+1 and map_states[i + locx][j + locy] > 0 and map_states[i + locx][j + locy] not in attackers:
                        # Send nearest units in logic way to cut off supply as defensive measure
                        # nearest_units = self.nearest_unit_to_space(friendly_unit_ids, unit_pos[self.player_idx], (i + locx, j + locy))
                        attackers.append(map_states[i + locx][j + locy])

        # Use attackers to dynamically move units?

        for i in range(total_units):
            # Prior to day 50? set up formation
            if self.day < 50:
                if friendly_unit_ids[i] in self.units_angle:
                    moves.append((1, self.units_angle[friendly_unit_ids[i]]))
                else:
                    self.units_angle[friendly_unit_ids[i]] = self.angles[i % len(self.angles)]
                    moves.append((1, self.angles[i % len(self.angles)]))
            # After day 50 switch to dynamic model
            else:
                if friendly_unit_ids[i] in offense_idx:
                    moves.append((1, offense[0]['mid_angle']) if (friendly_unit_ids[i] == offense[0]['move']) else ((1, offense[1]['mid_angle'])))
                else:
                    if friendly_unit_ids[i] in self.units_angle:
                        moves.append((1, self.units_angle[friendly_unit_ids[i]]) )
                    else:
                        self.units_angle[friendly_unit_ids[i]] = self.angles[i % len(self.angles)]
                        moves.append((1, self.angles[i % len(self.angles)]))

        self.day +=1
        return moves
