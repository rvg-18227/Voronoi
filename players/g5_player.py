import os
import pickle
import numpy as np
import sympy
import logging
from typing import Tuple
import sympy


def sympy_p_float(p: sympy.Point2D):
    return np.array([float(p.x), float(p.y)])

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
        self.spawn_days = spawn_days
        self.total_days = total_days
        self.num_days = 0
        self.naive_index = np.random.choice(2)

        if self.player_idx == 0:
            self.homebase = np.array([0.5, 0.5])
        elif self.player_idx == 1:
            self.homebase = np.array([0.5, 99.5])
        elif self.player_idx == 2:
            self.homebase = np.array([99.5, 99.5])
        else:
            self.homebase = np.array([99.5, 0.5])

        if self.player_idx == 0: # Top left
            self.initial_angles = [(4 * np.pi/8), 0]
        elif self.player_idx == 1: # Bottom left
            self.initial_angles = [3 * np.pi/2, 0] 
        elif self.player_idx == 2: # Bottom right
            self.initial_angles = [3 * np.pi/2, np.pi] 
        else: # Top right
            self.initial_angles = [np.pi, np.pi/2]
    
    def force_vec(self, p1, p2):
        v = p1 - p2
        mag = np.linalg.norm(v)
        if mag == 0:
            unit = v
        else:
            unit = v / mag
        return unit, mag

    def to_polar(self, p):
        x, y = p
        return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)

    def normalize(self, v):
        return v / np.linalg.norm(v)

    def repelling_force(self, p1, p2):
        dir, mag = self.force_vec(p1, p2)
        # Inverse magnitude: closer things apply greater force
        return dir * 1 / (mag)

    def attractive_force(self, p1, p2):
        return -self.repelling_force(p1, p2)
    

    def naive_strategy(self):
        angle = self.initial_angles[self.naive_index]
        distance = 1
        return distance, angle
        

    def border_strategy(self, unit_id, unit_pos, map_states, current_scores, total_scores, own_units, enemy_units_locations, mode, closest_border, borders_dist, offensive_arc_mean, attract_to_closest_border):
        border_x, border_y, border_center, border_dist = closest_border
        
        if mode == "defense":
            ALLY_INFLUENCE = -0.1
            ARC_MEAN_INFLUENCE = 0.3 # oldest 8 units, pushes defensive units
            # use this influence when we're > 20 units away from home base
           
        else:
            ALLY_INFLUENCE = 0.2
            ARC_MEAN_INFLUENCE = 0.0
        
        BORDER_INFLUENCE = 1
        
        ENEMY_INFLUENCE = 0.0

        if attract_to_closest_border:
            border_force = self.attractive_force(unit_pos, border_center)
        else:
            border_forces = [self.attractive_force(unit_pos, border_center) for x, y, border_center, border_dist in borders_dist]
            border_force = np.add.reduce(border_forces)
        
        
        if self.num_days // self.spawn_days > 8:
            arc_mean_force = self.attractive_force(unit_pos, offensive_arc_mean)
        else:
            arc_mean_force = np.array([0, 0])

        enemy_unit_forces = [
            self.attractive_force(unit_pos, enemy_pos)
            for _, enemy_pos in enemy_units_locations
        ]
        enemy_force = np.add.reduce(enemy_unit_forces)

        ally_force = [
            self.repelling_force(unit_pos, ally_pos)
            for ally_id, ally_pos in own_units
            if ally_id != unit_id and int(ally_id) % 3 == 0
        ]
        ally_force = np.add.reduce(ally_force)

        
        total_force = self.normalize(
            (arc_mean_force * ARC_MEAN_INFLUENCE)
            + (border_force * BORDER_INFLUENCE)
            + (enemy_force * ENEMY_INFLUENCE)
            + (ally_force * ALLY_INFLUENCE)
        )

        return self.to_polar(total_force)

    def is_in_map(self, pos):
        return 0 <= pos[0] < 100 and 0 <= pos[1] < 100

    def is_border_cell(self, x, y, map_states):
        top = (x, y - 1)
        bottom = (x, y + 1)
        left = (x - 1, y)
        right = (x + 1, y)
        top_left = (x - 1, y - 1)
        top_right = (x + 1, y - 1)
        bottom_left = (x - 1, y + 1)
        bottom_right = (x + 1, y + 1)
        neighbors = [top, bottom, left, right, top_left, top_right, bottom_left, bottom_right]
        for neighbor in neighbors:
            if self.is_in_map(neighbor):
                if map_states[neighbor[0]][neighbor[1]] != self.player_idx + 1:
                    return True
        return False



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
        self.num_days += 1

        own_units = list(
                zip(
                    unit_id[self.player_idx],
                    [sympy_p_float(pos) for pos in unit_pos[self.player_idx]],
                )
            )


        enemy_units_locations = [
            ((player, i), sympy_p_float(unit_pos[player][i]))
            for player in range(len(unit_pos))
            for i in range(len(unit_pos[player]))
            if player != self.player_idx
        ]



        borders = []
        for x in range(100):
            for y in range(100):
                if map_states[x][y] == self.player_idx + 1:
                   if self.is_border_cell(x, y, map_states):
                       borders.append((x, y))
        
        homebase_borders_dist = []
        for (x, y) in borders:
            border_center = np.array([x+0.5, y+0.5])
            homebase_borders_dist.append((x, y, border_center, np.linalg.norm(border_center - self.homebase)))
        homebase_borders_dist.sort(key=lambda x: x[3])
        if len(homebase_borders_dist) == 0:
            homebase_invasion = True
        elif homebase_borders_dist[0][3] < 20:
            homebase_invasion = True
        else:
            homebase_invasion = False

        moves = []

       
        border_assignment_set = set()
        
        offensive_arc_mean = np.mean([unit_pos for unit_id, unit_pos in own_units], axis=0)
        for i, (unit_id, unit_pos) in enumerate(own_units):
            if int(unit_id) % 3 == 0 and self.spawn_days < 20:
                # print("Unit id is ", unit_id)
                moves.append(self.naive_strategy())
            else:

                if homebase_invasion and len(own_units) - i <= 5:
                    attract_to_closest_border = True
                else:
                    attract_to_closest_border = False

                if i < len(own_units) // 4:
                    mode = "offense"
                else:
                    mode = "defense"
                
                borders_dist = []
                for (x, y) in borders:
                    border_center = np.array([x+0.5, y+0.5])
                    borders_dist.append((x, y, border_center, np.linalg.norm(border_center - unit_pos)))

                borders_dist.sort(key=lambda x: x[3])
                closest_border = None
                for (x, y, center, dist) in borders_dist:
                    if (x, y) not in border_assignment_set:
                        border_assignment_set.add((x, y))
                        closest_border = (x, y, center, dist)
                        break

                if closest_border is None:
                    if len(borders_dist) > 0:
                        closest_border = borders_dist[0]
                    else:
                        closest_border = (50, 50, np.array([50.0, 50.0]),
                                        np.linalg.norm(np.array([50.0, 50.0]) - unit_pos))

            
                moves.append(self.border_strategy(unit_id, unit_pos, map_states, current_scores, total_scores,
                        own_units, enemy_units_locations, mode, closest_border, borders_dist, offensive_arc_mean, attract_to_closest_border))
            
        return moves

