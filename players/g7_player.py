from ast import Num
from cmath import acos
from copy import deepcopy
from ctypes.wintypes import POINT
from doctest import DocFileSuite
import math
import os
import pickle
from re import L
from threading import currentThread
from turtle import distance
import numpy as np
import sympy
import logging
from typing import Tuple
from math import atan2, pi, dist
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Point


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
            self.angles = [np.deg2rad(46), np.deg2rad(0), np.deg2rad(72), np.deg2rad(18), np.deg2rad(90)]
        elif self.player_idx == 1: # Bottom left pink
            self.angles = [np.deg2rad(315), np.deg2rad(270), np.deg2rad(342), np.deg2rad(288), np.deg2rad(360)] 
        elif self.player_idx == 2: # Bottom right blue
            self.angles = [np.deg2rad(226), np.deg2rad(180), np.deg2rad(252), np.deg2rad(198), np.deg2rad(270)] 
        else: # Top right yellow
            self.angles = [np.deg2rad(135), np.deg2rad(90), np.deg2rad(162), np.deg2rad(108), np.deg2rad(180)]


        # home locations
        if self.player_idx == 0:
            self.locx = 0
            self.locy = 0
        elif self.player_idx == 1:
            self.locx = 0
            self.locy = 100
        elif self.player_idx == 2:
            self.locx = 100
            self.locy = 100
        elif self.player_idx == 3:
            self.locx = 100
            self.locy = 0

        self.home_point = Point(self.locx, self.locy)

        # Experimenal variables
        self.day = 0
        self.other_players = [0,1,2,3]
        self.angledef = []
        self.idstop = []
        self.other_players.remove(self.player_idx)
        self.last_n_days_positions = {}
        self.hostility_registry = {}
        # stores the units position, angle, and direction keyed by the units id
        self.unit_pos_angle = {}

        # stores a lists of unit ids keyed by their initial angle
        self.angles_taken = {}
        for angle in self.angles:
            self.angles_taken[angle] = []

        for other_player in self.other_players:
            self.last_n_days_positions[other_player] = []
            self.hostility_registry[other_player] = 1

    def find_attackers(self,map_states):
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
        return attackers

    # Find the nearest unit to a given space
    def nearest_units_to_unit(self, unit_position, unit_position_list, min_d=3):
        neighbor_dict = {'friendly': [], 'enemy':[]}
        # Iterate through all units to check for 
        for player in [0,1,2,3]:
            for unit in unit_position_list[player]:
                d = unit.distance(unit_position)
                # Assuming distance = 0 is the same unit
                if d <= min_d and d != 0.0:
                    if player == self.player_idx:
                        neighbor_dict['friendly'].append(unit)
                    else:
                        neighbor_dict['enemy'].append(unit)

        return [neighbor_dict['friendly'], neighbor_dict['enemy']]
        

    def nearest_enemy_space(self, unit_position, spaces):
        pos = (int(unit_position.x), int(unit_position.y))
        
        lowest_dir = None
        lowest_distance = 100

        directions = [
            [0, -1], 
            [1, -1], 
            [1, 0],   
            [1, 1],    
            [0, 1],    
            [-1, 1],  
            [-1, 0], 
            [-1, -1], 
        ]
        target_space = 232
        # Iterate in each direction and find the closest enemy space
        for dir in directions:
            # Try every step length between 1 and the current lowest distance
            for i in range(1, lowest_distance):
                mod_x = pos[0]+i*dir[0]
                mod_y = pos[1]+i*dir[1]

                if mod_x > 99 or mod_x < 0: # If the x is off the map stop trying this direction
                    break
                elif mod_y > 99 or mod_y < 0: # If the y is off the map stop trying this direction
                    break
                
                if spaces[mod_x][mod_y] != (self.player_idx+1): # If the space is not our space it is the closest so save it and the direction
                    lowest_dir = dir
                    lowest_distance = i
                    break

        return(np.arctan2(lowest_dir[1], lowest_dir[0]), lowest_distance)


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
    
    def closest_t(self, current_unit_pos, teammates, team_distance, closest_teammate_dist, closest_teammate):
        for t in teammates:
            if t == current_unit_pos:
                d = 0
            else:
                d = t.distance(current_unit_pos)
            # on top of eachother will throw error
            if d != 0:
                team_distance += 1/d
            if d < closest_teammate_dist:
                closest_teammate_dist = d
                closest_teammate = t
        return team_distance, closest_teammate_dist, closest_teammate
    
    def closest_e(self, current_unit_pos, enemies, enemy_distance, closest_enemy_dist, closest_enemy):
        for e in enemies:
            if e == current_unit_pos:
                d = 0
            else:
                d = e.distance(current_unit_pos)
            # on top of eachother will throw error
            if d != 0:
                enemy_distance += 1/d
            if d < closest_enemy_dist:
                closest_enemy_dist = d
                closest_enemy = e
        return enemy_distance, closest_enemy_dist, closest_enemy
        
    def behavior(self, closest_teammate, closest_enemy, unit, current_unit_pos, direction_empty_space, closest_teammate_dist, closest_enemy_dist):
        dist = 0
        angle = 0
        # From here would like to see 4 core behaviors

        # 1. no enemies around -> move towards combination of away from teamates and towards empty space
            # Speed should be 1 here I think basically no matter what
            # Direction can be as simple as combination of those two angles
        if not closest_enemy and closest_teammate:
            diff_x, diff_y = closest_teammate.x-current_unit_pos.x, closest_teammate.y-current_unit_pos.y
            theta_team = atan2(diff_y, diff_x)

            # Go in opposite direction
            if theta_team < 0:
                theta_team+=pi
            else:
                theta_team-=pi

            return theta_team, 1

        # 2. nothing around -> move towards empty space or if close to home move in principle direction
            # Speed should be 1
            # direction given by direction_empty_space
        elif not closest_enemy and not closest_teammate:
            angle = direction_empty_space
            dist = 1
            return angle, dist
        
        # 3. only enemies around -> move in combination of opposite from enemies and towards home
            # Speed towards home should depend on distance from home and distance from enemies
            # As we get closer to enemies with no support move away faster
            # As we get further from home move faster aswell
        elif closest_enemy and not closest_teammate:
            angle = atan2(self.home_point.x - current_unit_pos.x, self.home_point.y - current_unit_pos.y)
            if current_unit_pos.distance(self.home_point) < closest_enemy_dist and closest_teammate_dist > closest_enemy_dist:
                dist = .5
            else: 
                dist = 1
            return angle, dist

        # 4. enemies and teamates around -> use computed distances from team and enemies to pincer or stay still
            # Speed based on saftey metrics again
            # offset angle to get around enemies if we have more support than danger
        else:
            angle = direction_empty_space
            dist = 1
            return angle, dist
            
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
        moves = []

        total_friendly_units = len(unit_id[self.player_idx])
        friendly_unit_ids = unit_id[self.player_idx] 

        # Remove Dead Units
        dead_units = [unit for unit in self.unit_pos_angle if unit not in friendly_unit_ids]
        for dead_unit in dead_units:
            # remove from unit list
            del self.unit_pos_angle[dead_unit]

            # remove from principle angle index
            for (angle,list_units) in self.angles_taken.items():
                if dead_unit in list_units:
                    list_units.remove(dead_unit)

        # Add new units to dictionary and update locations of old ones
        for i in range(len(unit_id[self.player_idx])):
            # If this is a new unit
            if friendly_unit_ids[i] not in self.unit_pos_angle:
                # Decide angle by finding the principle angle with the least number of units
                minumum_length = 100
                minumum_key = -1
                for (angle,list_units) in self.angles_taken.items():
                    if len(list_units) < minumum_length:
                        minumum_key = angle
                        minumum_length = len(list_units)

                        # If there is no unit with a principle angle in this direction we can simple choose this angle
                        if minumum_length == 0:
                            break

                # Add the unit to both lists
                self.angles_taken[minumum_key].append(friendly_unit_ids[i])
                self.unit_pos_angle[friendly_unit_ids[i]] = {'angle': minumum_key, 'distance':1}

            # Update position
            self.unit_pos_angle[friendly_unit_ids[i]]['pos'] = unit_pos[self.player_idx][i]

        ### previous strats
        # attackers = self.find_attackers(map_states)
        # offense = self.moveTowardAggressive(current_scores, unit_pos, unit_id)
        # offense_idx = [i['move'] for i in offense]
        ###

        # If the day is greater than 50 formation building is over enter dynamic template    
        if self.day > 50:
            queue = friendly_unit_ids
            np_spaces = np.array(map_states)
            for unit in queue:
                
                #initializing closest unit types
                closest_teammate = False
                closest_enemy = False
                closest_teammate_dist = 100
                closest_enemy_dist = 100
                
                # Find angle to move away from nearby unit if friendly and move around unit if enemy
                team_distance = 0
                enemy_distance = 0
                
                current_unit_pos = self.unit_pos_angle[unit]['pos']

                # Find direction and distance to nearest space, need to normalize and test direction
                [direction_empty_space, distance_empty_space] = self.nearest_enemy_space(current_unit_pos, np_spaces)

                #    continue # Continue moving this unit in principle direction since there are no enemies nearby
                if distance_empty_space > 10:
                    continue
                
                # Find all nearby units
                [teammates, enemies] = self.nearest_units_to_unit(current_unit_pos, unit_pos)

                # Find the total inverse distance from teamates and also the closest teamate 
                team_distance, closest_teammate_dist, closest_teammate = self.closest_t(current_unit_pos, teammates, team_distance, closest_teammate_dist, closest_teammate)

                # Find the total inverse distance from enemies and also the closest enemy 
                enemy_distance, closest_enemy_dist, closest_enemy = self.closest_e(current_unit_pos, enemies, enemy_distance, closest_enemy_dist, closest_enemy)

                self.unit_pos_angle[unit]['angle'] = direction_empty_space
                self.unit_pos_angle[unit]['distance'] = 1
                
                self.unit_pos_angle[unit]['angle'], self.unit_pos_angle[unit]['distance'] = self.behavior(closest_teammate, closest_enemy, unit, current_unit_pos, direction_empty_space, closest_teammate_dist, closest_enemy_dist)
                
        for i in range(total_friendly_units):
            moves.append((self.unit_pos_angle[friendly_unit_ids[i]]['distance'], self.unit_pos_angle[friendly_unit_ids[i]]['angle']))


        self.day +=1
        return moves