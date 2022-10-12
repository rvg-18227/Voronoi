import os
import pickle
from turtle import width
#from time import clock_settime
import numpy as np
import sympy
import logging
from typing import Tuple
from sympy.geometry import Point2D
from enum import Enum
from scipy.ndimage import measurements, morphology
import math
import random

class UnitType(Enum):
    SPACER = 0
    ATTACK = 1
    DEFENSE = 2

class Defense:

    def __init__(self, player_idx, spawn_point):
        self.unitType = UnitType.DEFENSE
        self.number_units = 0
        self.prev_state = None
        self.spawn_point = (spawn_point.x, spawn_point.y)
        self.player_idx = player_idx
        self.day = 0
        self.scanner_radius = 20

    def update(self, map_state, defenderIdxs, units, enemy_units):
        self.map_state = map_state
        #rotate the map state to the bottom left
        self.number_units = len(defenderIdxs)
        self.defenderIdxs = defenderIdxs
        self.unit_locations = [unit for i, unit in enumerate(units) if i in defenderIdxs]
        self.enemy_units = enemy_units
        self.day += 1

    def get_moves(self):

        moves = [(0, 0, 0) for i, pos in enumerate(self.unit_locations)]
        moved = [False for _ in range(self.number_units)]
        if self.number_units == 0:
            return []


        clusters = self.get_clusters()
        units_left_to_allocate = self.number_units
        clusters_to_defend = 0

        # Determine how many clusters to defend
        for i, cluster in enumerate(clusters):
            if len(cluster["points"]) < units_left_to_allocate:
                units_left_to_allocate -= (len(cluster["points"]) + 1)
                clusters_to_defend += 1
            else:
                break

        clusters = clusters[:clusters_to_defend]
        clusters = sorted(clusters, key=lambda x: x["angle"])

        # for each unit, allocate it to the closest cluster
        cluster_points_left = [len(cluster["points"]) for cluster in clusters]
        for i, unit in enumerate(self.unit_locations):
            distances = sorted([(idx, np.linalg.norm(np.array(unit) - cluster["centroid"])) for idx, cluster in enumerate(clusters)], key=lambda x: x[1])

            for j, (idx, distance) in enumerate(distances):
                if distance > self.scanner_radius:
                    break
                offset_weight = 3
                if cluster_points_left[idx] == 0:# and np.linalg.norm(self.spawn_point - clusters[idx]["centroid"]) < 60:
                    offset_weight = 4

                if cluster_points_left[idx] > 0 or offset_weight != 3:
                    cluster_points_left[idx] -= 1
                    moved[i] = True
                    if cluster_points_left[idx] < 0:
                        cluster_point_distances = [np.linalg.norm(np.array(self.spawn_point) - np.array(point)) for idx, point in enumerate(clusters[idx]["points"])]
                        min_dist = max(cluster_point_distances).item()
                        idx_of_closest_cluster_point = cluster_point_distances.index(min_dist)
                    target_point = clusters[idx]["points"][cluster_points_left[idx] if cluster_points_left[idx] >= 0 else idx_of_closest_cluster_point]
                    target_point = np.floor(target_point)
                    target_point += np.array((0.5, 0.5))

                    direction = target_point[0] - unit.x, target_point[1] - unit.y

                    goal_direction = 0 if unit.x < 20 else self.spawn_point[0] - unit.x, 0 if unit.y < 20 else self.spawn_point[1] - unit.y
                    offset = (0, 0) if np.linalg.norm(goal_direction) == 0 else goal_direction/np.linalg.norm(goal_direction)

                    #TODO: if formation is ready, offset = 0
                    if self.number_in_circle(self.unit_locations, unit, 1) > self.number_in_circle(self.enemy_units, target_point, 1):
                        offset_weight -= 3

                    end_direction = direction + offset*offset_weight
                    distance_to_goal = np.linalg.norm(end_direction)

                    moves[i] = distance_to_goal, end_direction[0], end_direction[1]
                    break
            # move adding the offset here
            # for loop
            # if cluster not all true
            # add offset

        n_free_units = self.number_units - sum(moved)
        hover_points = self.get_hover_points(n_free_units)
        offset = (1/6)
        hover_points = [np.array(point) + (self.spawn_point - np.array(point))*offset for point in hover_points]

        free_units = [i for i, move in enumerate(moves) if not moved[i]]

        for point in hover_points:
            distances = sorted([(idx, np.linalg.norm(np.array(unit) - point)) for idx, unit in enumerate(self.unit_locations)], key=lambda x: x[1])
            for j, (idx, distance) in enumerate(distances):

                if idx in free_units:
                    free_units.remove(idx)
                    moves[idx] = distance, point[0] - self.unit_locations[idx].x, point[1] - self.unit_locations[idx].y
                    break

        # if path to homme width is < some X, retreat to hover point
        
        # make units the reverse of the cluster (negate then add 2*(closest unit to 0)) - some offset towards home
        # once all units are in place - (current loc to calc place ~=, for all units in this one match)
        # self.prev_state = self.map_state

        return moves

    def get_hover_points(self, n):
        hover_points = []
        if n == 0:
            n = 1
        step = 90/n

        degrees_to_hover = [(step*i) + 90/(n+1) for i in range(n)]
        random.shuffle(degrees_to_hover)

        for deg in degrees_to_hover:
            angle = math.radians(deg)
            angle = angle - (math.pi/2 * self.player_idx)
            hover_points.append(self.get_raycast_to_border(angle))

        return hover_points
    
    def get_raycast_to_border(self, angle):
        min_dist = 25
        max_dist = 75
        step = 1
        for i in reversed(range(min_dist, max_dist, step)):
            point = self.spawn_point + np.array((i*math.cos(angle), i*math.sin(angle)))
            if point[0] > 100 or point[0] < 0 or point[1] > 100 or point[1] < 0:
                continue
            if self.map_state[math.floor(point[0])][math.floor(point[1])] == self.player_idx+1:
                return point

        return self.spawn_point

    def number_in_circle(self, units, center, radius):
        units = set([tuple(np.floor(unit)) for unit in units])
        units.add(tuple(np.floor(center)))
        return sum([1 if np.linalg.norm(np.floor(np.array(unit)) - np.floor(center)) <= radius else 0 for unit in units])
    
    def get_clusters(self):
        """Returns a list of clusters of points"""  
        floored_enemy = [(np.floor(unit.x), np.floor(unit.y)) for unit in self.enemy_units]
        binary_enemy_unit_map = [[1 if (i, j) in floored_enemy else 0 for j, _ in enumerate(row)] for i, row in enumerate(self.map_state)]
        lw, num = measurements.label(binary_enemy_unit_map)
        clusters = [{
            "points": []
        } for _ in range(num)]
        num_added = 0

        for i, row in enumerate(lw):
            for j, cell in enumerate(row):
                if cell != 0:
                    clusters[cell-1]["points"].append((i, j))
                    num_added += 1

        clusters = [cluster for cluster in clusters if len(cluster["points"]) > 0]

        for cluster in clusters:
            cluster["centroid"] = np.mean(cluster["points"], axis=0)
            cluster["distance"] = np.linalg.norm(cluster["centroid"] - self.spawn_point)
            cluster["angle"] = np.arctan2(cluster["centroid"][1] - self.spawn_point[1], cluster["centroid"][0] - self.spawn_point[0])

        clusters = sorted(clusters, key=lambda x: x["distance"])
        return clusters

class Attacker:

    def __init__(self, id, position):
        self.unitType = UnitType.ATTACK
        self.number_units = 0
        self.prev_state = None
        self.player_id = id
        self.x = float(position.x)
        self.y = float(position.y)
        self.day = 0
        self.real_ids = []
        
        self.seen = []
        self.lr_counter = 0

        self.left_list = []
        self.right_list = []
        self.tmp_left_list = []
        self.tmp_right_list = []
        self.final_left_list = []
        self.final_right_list = []
        



    def update(self, map_state, attackerIdxs, units, enemy_units, unit_id):
        self.map_state = map_state
        #rotate the map state to the bottom left
        self.number_units = len(attackerIdxs)
        self.attackerIdxs = attackerIdxs
        self.unit_id = unit_id
        self.real_ids = self.unit_id[self.player_id]

        self.unit_locations = [(unit,i) for i, unit in enumerate(units) if i in attackerIdxs]


        for i, (unit,pos) in enumerate(self.unit_locations):
            real_id = self.real_ids[pos]

            if real_id not in self.seen:
                self.lr_counter += 1
                self.seen.append(real_id)

                if 1 <= self.lr_counter <= 3:    
                    self.left_list.append(real_id)
                elif 4 <= self.lr_counter <= 6:
                    self.right_list.append(real_id)
                    if self.lr_counter == 6:
                        self.lr_counter = 0


        self.enemy_units = enemy_units
        self.day += 1

    def get_moves(self):
        moves = [(0, 0, 0) for i, (unit,pos) in enumerate(self.unit_locations)]
        moved = [False for _ in range(self.number_units)]
        if self.number_units == 0:
            return []


        for i, (unit,pos) in enumerate(self.unit_locations):
            real_id = self.real_ids[pos]
            triplet_no = (self.seen.index(real_id)) % 3 # Is the unit 1, 2, or 3 in the triplet to determine special movement

            # # Code to group up triplets
            # if real_id in self.left_list and real_id not in self.final_left_list:
            #     if triplet_no == 0 or triplet_no == 1:
            #         moves[i] = 1, 1, 0 # Move left
            #         self.tmp_left_list.append(real_id)
            #         self.left_list.remove(real_id)
            #     else:
            #         moves[i] = 1, 1, 0
            #         self.left_list.remove(real_id)
            #         self.final_left_list.append(real_id)
            #         self.final_left_list.extend(self.left_list)
            #         self.left_list.clear()

            # elif real_id in self.final_left_list:
            #     moves[i] = 1, 1, 0 


            # elif real_id in self.right_list and real_id not in self.final_right_list:
            #     if triplet_no == 0 or triplet_no == 1:
            #         moves[i] = 1, 0, 1 # Move right
            #         self.tmp_right_list.append(real_id)
            #         self.right_list.remove(real_id)
            #     else:
            #         moves[i] = 1, 0, 1
            #         self.right_list.remove(real_id)
            #         self.final_right_list.append(real_id)
            #         self.final_right_list.extend(self.right_list)
            #         self.right_list.clear()

            # elif real_id in self.final_right_list:
            #     moves[i] = 1, 0, 1 



            if real_id in self.left_list:
                # Even - LEFT attacking troops
                moves[i] = 1, 1, 0 
            elif real_id in self.right_list:
                # Odd - RIGHT attacking troops
                moves[i] = 1, 0, 1 #distance_to_goal, end_direction[0], end_direction[1]  - Pos x only (right)

            else:
                moves[i] = 0, 0, 1 # Does not move


        return moves






class Spacer:
    
    def __init__(self, player_idx, spawn_point):
        self.unitType = UnitType.SPACER
        self.number_units = 0
        self.prev_state = None
        self.spawn_point = spawn_point
        self.player_idx = player_idx
        self.day = 0
        self.scanner_radius = 50

    def update(self, map_state, spacerIdxs, units, enemy_units):
        self.map_state = map_state
        #rotate the map state to the bottom left
        self.number_units = len(spacerIdxs)
        self.spacerIdxs = spacerIdxs
        self.unit_locations = [unit for i, unit in enumerate(units) if i in spacerIdxs]
        self.enemy_units = enemy_units
        self.day += 1
    def get_moves(self):
        moves = {}
        if self.number_units == 0:
            return moves
        #print()
        # Adapted from group 4 Code
        ENEMY_INFLUENCE = 1 # TODO: change influence values for best spacers!
        HOME_INFLUENCE = 20
        ALLY_INFLUENCE = 0.5

        for i, unit in enumerate(self.unit_locations):
            unit = np.array(unit)
            #print('u', unit)
            enemy_force = np.add.reduce([self.repelling_force(unit,np.array(enemy)) for enemy in self.enemy_units])
            #print('sp',(self.spawn_point))
            home_force = self.repelling_force(unit, np.array(self.spawn_point))
            #print('hghgh',enemy_force, home_force)
            ally_forces_list = []
            for ally in self.unit_locations:
                ally = np.array(ally)
                #print('ally',ally)
                if not np.array_equal(ally, unit): 
                    #print('not ally')
                    ally_forces_list.append(self.repelling_force(unit,ally))
            ally_force = np.add.reduce(ally_forces_list)
            #ally_force = np.add.reduce([self.repelling_force(unit,ally) for ally in self.unit_locations if not unit]) ###TODO: currently only looks at other SPACERS as ally's not all. Need to change this.
            #print(('ally force:', ally_force))
            total_force = self.normalize((enemy_force * ENEMY_INFLUENCE)
                                         + (home_force * HOME_INFLUENCE)
                                         + (ally_force * ALLY_INFLUENCE)
                                         )
            #print(unit)
            #print(type(unit))
            #print(total_force)
            moves[i] = self.to_polar(total_force)
        #print('moves',moves)
        return moves

    def force_vec(self, p1, p2):
        v = p1 - p2
        mag = np.linalg.norm(v)
        #print(v,'over',mag)
        unit = v / mag
        return unit, mag

    def to_polar(self, p):
        x, y = p
        return np.sqrt(x**2 + y**2), np.arctan2(y, x)

    def normalize(self, v):
        return v / np.linalg.norm(v)

    def repelling_force(self, p1, p2):

        if np.array_equal(p1,p2): 
            return 0
        dir, mag = self.force_vec(p1, p2)
        return dir * 1 / (mag)        




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

        testing = False
        if testing:
            testingType = UnitType.DEFENSE
            self.PHASE_ONE_OUTPUT = [testingType]
            self.PHASE_TWO_OUTPUT = [testingType]
            self.PHASE_THREE_OUTPUT = [testingType]
        #change these based on how much land we have? aka if more ppl are targeting us vs less?
        # if less land it means our defense is losing so we need more
        # if more land it means we want to attack more?


        self.number_units_total = total_days//spawn_days
        self.PHASE_ONE_UNITS = 5
        self.PHASE_TWO_UNITS = int(np.floor((self.number_units_total-5)*0.6))
        self.PHASE_THREE_UNITS = self.number_units_total - self.PHASE_ONE_UNITS - self.PHASE_TWO_UNITS

        #locations of units
        self.unit_types = {
            UnitType.SPACER: {},
            UnitType.ATTACK: {},
            UnitType.DEFENSE: {}
        }

        self.spacer = Spacer(self.player_idx, self.spawn_point)
        self.defense = Defense(self.player_idx, self.spawn_point)
        self.attack = Attacker(self.player_idx, self.spawn_point)

    def play(self, unit_id, unit_pos, map_states, current_scores, total_scores) -> [Tuple[float, float]]:
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
        self.add_spawn_units_if_needed(unit_id[self.player_idx])
        spacers, attackers, defenders = self.get_unit_indexes(unit_id[self.player_idx]) 

        enemy_ids, enemy_units = self.get_enemy_units(unit_id, unit_pos)

        # 3 roles
        # attackers - Identify weak enemy, how to kill a unit? where to attack? when to attack? hover at border until enough units? whats the best formation?
        # defenders - kmeans number of units around/units in a radius, most important!!!, so be able to use other groups if needed
        # space gain people - look at group 4's code or come up wiht a new generic strategy

        # 3 phases
        # in intro phase allocate X spacegain until 5 units?
        # in main phase allocate X attackers and Y defenders and Z spacegain 2:2:1, maybe change ratio based on if we're being attacked or not
        # in end phase allocate X attackers and Y defenders in some proportion 

        self.map_states = map_states
        self.unit_pos = unit_pos
        self.left_right_count = 0

        moves = [self.transform_move(0, 0, 0)] * len(unit_pos[self.player_idx])

        self.spacer.update(self.map_states, spacers, unit_pos[self.player_idx], enemy_units)
        #print(spacers)
        #print('self.spacer.number_units',self.spacer.number_units)
        spacerMoves = self.spacer.get_moves()
        #print('spacerMoves',spacerMoves)
        
        for spacermoveidx, real_idx in enumerate(spacers):
            moves[real_idx] = spacerMoves[spacermoveidx]

        self.attack.update(self.map_states, attackers, unit_pos[self.player_idx], enemy_units, unit_id)
        attackingMoves = self.attack.get_moves()
        for attacking_move_idx, real_idx in enumerate(attackers):
            dist, x, y = attackingMoves[attacking_move_idx]
            moves[real_idx] = (dist if dist <= 1 else 1, np.arctan2(y, x))

        self.defense.update(self.map_states, defenders, unit_pos[self.player_idx], enemy_units)
        defensiveMoves = self.defense.get_moves()
        for defensive_move_idx, real_idx in enumerate(defenders):
            dist, x, y = defensiveMoves[defensive_move_idx]
            moves[real_idx] = (dist if dist <= 1 else 1, np.arctan2(y, x))

        self.current_turn += 1
        return moves

    def simulate_move(self, unit_pos, move) -> Tuple[float, float]:
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

    def transform_move(self, x: float, y: float, distance) -> Tuple[float, float]:
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

        for i, unit_id in enumerate(unit_ids):
            if unit_id in self.unit_types[UnitType.SPACER]:
                self.unit_types[UnitType.SPACER][unit_id] = i 
            elif unit_id in self.unit_types[UnitType.ATTACK]:
                self.unit_types[UnitType.ATTACK][unit_id] = i
            elif unit_id in self.unit_types[UnitType.DEFENSE]:
                self.unit_types[UnitType.DEFENSE][unit_id] = i

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

    def get_enemy_units(self, unit_id, unit_pos):
        """Returns list of coords of enemy units
                Args:
                    unit_id (list(int)): list of unit ids
                    unit_pos (list(Point2D)): list of unit positions
                Returns:
                    Tuple[list(int), list(Point2D)]: indexes of the enemy units and coords of the enemy units
        """
        enemy_unit_ids = []
        enemy_unit_pos = []

        for i in range(len(unit_pos)):
            if i != self.player_idx:
                enemy_unit_ids += unit_id[i]
                enemy_unit_pos += unit_pos[i]

        return enemy_unit_ids, enemy_unit_pos
