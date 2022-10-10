import os
import pickle
from matplotlib.pyplot import close
import numpy as np
import sympy
import logging
from typing import Tuple, List
import math

from scipy.spatial.distance import cdist

#HYPER PARAMETERS
DANGER_ZONE_RADIUS = 20

#Dictionary into 2d Array
def points_to_numpy(units):

    #columns: x, y, player_idx belongs, 
    result = np.array([0,0,0])

    for i in range(4):
        for u in units[i]:
            result = np.vstack([result, [u.x, u.y, i]])

    return result[1:,:]


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

    def transform_move (self, dist_ang: Tuple[float, float]) -> Tuple[float, float]:
        dist, rad_ang = dist_ang
        return (dist, rad_ang - (math.pi/2 * self.player_idx))


    def play(self, unit_id, unit_pos, map_states, current_scores, total_scores) -> List[Tuple[float, float]]:
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
        

        if self.player_idx == 1:

            #for the first 50 days
            #first unit goes straight diagonal (45 deg)
            #second unit goes at a 22.5 degree
            #third unit goes at a 67.5 degree

            #same dimension as unit_pos
            danger_levels = self.danger_levels(unit_pos)

            moves = []
            scout_targets = [(50,20), (20,50)]

            danger_levels_ally = danger_levels[self.player_idx]

            frontier = [[0],[1],[2]]

            for i in range(len(unit_id[self.player_idx])):

                curr_x = unit_pos[self.player_idx][i].x
                curr_y = unit_pos[self.player_idx][i].y

                # first unit holds 50,50
                if i == 0:
                    distance = 1

                    angle = (45)

                    moves.append((distance, angle*(math.pi / 180)))

                #scouts the left and right wing
                elif i == 1:
                    distance = min(1, math.dist([curr_y,curr_x],scout_targets[0]))
                    distance = 1
                    angle = 22.5

                    moves.append((distance, angle*(math.pi / 180)))
                
                elif i == 2:
                
                    #distance = min(1, math.dist([curr_y,curr_x],scout_targets[1]))
                    distance = 1
                    angle = 67.5

                    moves.append((distance, angle*(math.pi / 180)))

                else:
                    moves.append((1, 45*(math.pi / 180)))

            #print(moves)
            return [self.transform_move(move) for move in moves]

        else:
            return self.sprinkler_player(unit_id)

    #First Algorithm
    def sprinkler_player(self, unit_id):
        moves = []
        angle_jump = 10
        angle_start = 45
        for i in range(len(unit_id[self.player_idx])):
            distance = 1

            angle = (((i) * (angle_jump) + angle_start ))%90

            moves.append((distance, angle* (math.pi / 180)))

        return [self.transform_move(move) for move in moves]

    #start claculating only after day 40+
    #unlikely to have nearby enemies day 40
    def danger_levels(self, unit_pos):

        np_unit_pos = points_to_numpy(unit_pos)

        result = [[] for _ in range(4)]

        for team_idx in range(4):
            units = unit_pos[team_idx]

            for u in units:
                distances = cdist([np.array((u.x, u.y))], np_unit_pos[:, :2]).flatten()
                close_points = np_unit_pos[distances <= DANGER_ZONE_RADIUS,:]
                team_points = close_points[close_points[:,2] == team_idx]

                #should always be >=1 because it counts itself
                close_team_count = team_points.shape[0]

                #total count - team count = enemy count
                close_enemy_count = close_points.shape[0]-close_team_count

                #print("TeamCount: ", close_team_count)
                #print("EnemyCount: ", close_enemy_count)

                danger_score = close_enemy_count/close_team_count
                result[team_idx].append(danger_score)
                
        #print(result)
        return result

