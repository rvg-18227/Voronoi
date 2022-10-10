from collections import defaultdict
import os
import pickle
from matplotlib.pyplot import close
import numpy as np
import sympy
import logging
from typing import Tuple, List, Dict
import math
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from scipy.spatial.distance import cdist
import heapq

#HYPER PARAMETERS
DANGER_ZONE_RADIUS = 20

class InterestRegion:
    def __init__(self, center_point, polygon, created_idx):
        self.center_point: Tuple[float, float] = center_point
        self.polygon: Polygon = polygon
        self.created_idx: float = created_idx

    def __hash__(self):
        return hash(self.center_point)

    def __lt__(self, other):
        return self.created_idx < other.created_idx

    def __repr__(self):
        return(str(self.polygon))

#Dictionary into 2d Array
def points_to_numpy(units):
    #columns: x, y, player_idx belongs, 
    result = np.array([0,0,0])

    for i in range(4):
        for u in units[i]:
            result = np.vstack([result, [u.x, u.y, i]])

    return result[1:,:]

def get_corner(idx):
    if idx == 0:
        return (0,0)
    elif idx == 1:
        return (0,100)
    elif idx == 2:
        return (100, 100)
    elif idx == 3:
        return (100, 0)

def get_interest_regions(idx) -> List[InterestRegion]:
    x, y = get_corner(idx)
    regions = []

    regions.append(get_interest_regions_points((50,50),0))

    if x == 0:
        regions.append(get_interest_regions_points((10,50),1))
        regions.append(get_interest_regions_points((30,50),2))
    elif x == 100:
        regions.append(get_interest_regions_points((90,50),1))
        regions.append(get_interest_regions_points((70,50),2))

    if y == 0:
        regions.append(get_interest_regions_points((50,10),3))
        regions.append(get_interest_regions_points((50,30),4))
    elif y == 100:
        regions.append(get_interest_regions_points((50,90),3))

        regions.append(get_interest_regions_points((50,70),4))

    return regions

def get_interest_regions_points(center_point, i):
    x, y = center_point
    return InterestRegion(center_point, Polygon([(x-10, y-10), (x+10, y-10), (x+10, y+10), (x-10,y+10)]), i)

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
        self.days = 1

        self.regions = get_interest_regions(player_idx)
        self.otw_to_regions = {}
        self.region_otw = defaultdict(lambda: set())

    def transform_move (self, dist_ang: Tuple[float, float]) -> Tuple[float, float]:
        dist, rad_ang = dist_ang
        return (dist, rad_ang - (math.pi/2 * self.player_idx))

    def point_move(self, p1, p2):
        dist = min(1, math.dist(p1,p2))
        angle = sympy.atan2(p2[1] - p1[1], p2[0] - p1[0])
        return (dist, angle)
    
    def fixed_formation_moves(self, unit_ids, angles, move_size=1):
        return {
            int(unit_id) : 
                self.transform_move((move_size, angles[idx % len(angles)]*(math.pi / 180)))
            for idx, unit_id
            in enumerate(unit_ids)
        }
    
    def get_home_coords(self):
        if self.player_idx == 0:
            return Point(0.5, 0.5)
        elif self.player_idx == 1:
            return Point(0.5, 99.5)
        elif self.player_idx == 2:
            return Point(99.5, 99.5)
        elif self.player_idx == 3:
            return Point(99.5, 0.5)

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

        # Initialize all unit moves to null so that an unspecified move is equivalent to not moving
        moves = {int(id): None for id in unit_id[self.player_idx]}

        # Send fixed scouts to capture early game wins
        moves.update(self.fixed_formation_moves(unit_id[self.player_idx][0:3], [45.0, 67.5, 22.5]))
 
        #if all the moves are fixed formation moves:
        if len(moves) <= 3:
            return list(moves.values())

        danger_levels, danger_regions_score, region_count, unit_regions = self.danger_levels(unit_pos)

        region_and_score = []

        #Move to quadrant with (in priority, highest first):
        #no units > danger score > closest
        for r in self.regions:
            temp = []

            if r in region_count:
                temp.append(region_count[r])
            else:
                temp.append(0)

            temp[0] += len(self.region_otw[r])

            if r in danger_regions_score:
                if danger_regions_score[r] != 0:
                    temp.append(-danger_regions_score[r])
                else:
                    temp.append(-float("inf"))
            else:
                temp.append(-float("inf"))

            temp.append(r)
            heapq.heappush(region_and_score,tuple(temp))

        for i in range(len(unit_id[self.player_idx])):
            
            pt = unit_pos[self.player_idx][i]
            curr = (pt.x, pt.y)

            if i in unit_regions:
                self.otw_to_regions.pop(i, None)
                self.region_otw[unit_regions[i]].discard(i)
                moves[i+1] = self.point_move(curr, unit_regions[i].center_point)
                continue

            if i in self.otw_to_regions:
                moves[i+1] = self.point_move(curr, self.otw_to_regions[i])
                continue

            target = list(heapq.heappop(region_and_score))
            #print(region_and_score)

            moves[i+1] = self.point_move(curr, target[2].center_point)

            self.otw_to_regions[i] = target[2].center_point
            print(i, target[2].center_point)
            self.region_otw[target[2]].add(i)

            heapq.heappush(region_and_score, tuple([target[0]+1, target[1], target[2]]))

        self.days += 1  

        #print(list(moves.values()))  
        print(moves)
        return list(moves.values())

    def danger_levels(self, unit_pos):
        np_unit_pos = points_to_numpy(unit_pos)
        result = [[] for _ in range(4)]
        danger_regions = {}
        region_count = {}
        unit_regions = {}

        for team_idx in range(4):
            for unit_idx, u in enumerate(unit_pos[team_idx]):
                distances = cdist([np.array((u.x, u.y))], np_unit_pos[:, :2]).flatten()
                close_points = np_unit_pos[distances <= DANGER_ZONE_RADIUS,:]
                team_points = close_points[close_points[:,2] == team_idx]

                #should always be >=1 because it counts itself
                close_team_count = team_points.shape[0]

                close_enemy_count = close_points.shape[0]-close_team_count

                danger_score = close_enemy_count/close_team_count
                result[team_idx].append(danger_score)

                #check if in any region
                for region in self.regions:
                    if region.polygon.contains(Point(u.x, u.y)):
                        if team_idx == self.player_idx:
                            unit_regions[unit_idx] = region

                            if region in danger_regions:
                                danger_regions[region] += danger_score
                            else:
                                danger_regions[region] = danger_score

                            if region in region_count:
                                region_count[region] += 1
                            else:
                                region_count[region] = 1

                        continue

        #result is the danger level of all units (same dimensions as unit_pos)

        #danger_regions is a dict with {(center_point of region) : sum of danger_level of teammate}

        #region_count is number of teammates in region

        #unit_regions is which region is a unit in
        return result, danger_regions, region_count, unit_regions

    def wall_forces(self, current_point) -> Dict[Tuple[float, float], float]:
        current_x = current_point.x
        current_y = current_point.y
        dist_to_top = Point(current_x, 0).distance(current_point)
        dist_to_bottom = Point(current_x, 100).distance(current_point)
        dist_to_right = Point(0, current_y).distance(current_point)
        dist_to_left = Point(100, current_y).distance(current_point)
        return {(50, 0): dist_to_top, (50, 0): dist_to_bottom, (100, 50): dist_to_right, (0, 50): dist_to_left}

    def closest_friend_force(self, current_unit, current_pos, unit_pos, unit_id) -> Dict[Tuple[float, float], float]:
        if(len(unit_id[self.player_idx]) < 2):
            return None

        closest_unit_dist = math.inf
        for i in range(len(unit_pos[self.player_idx])):
            if i == current_unit:
                continue
            friend_unit = unit_id[self.player_idx][i]
            friend_unit_pos = unit_pos[self.player_idx][i]
            dist = friend_unit_pos.distance(current_pos)
            if dist < closest_unit_dist:
                closest_unit_dist = dist
        return {friend_unit_pos: closest_unit_dist}        

    def get_forces(self, unit_id, unit_pos, danger_regions):
        forces = {id: {} for id in unit_id[self.player_idx]}
        for i in range(len(unit_id[self.player_idx])):
            unit = unit_id[self.player_idx][i]
            current_pos = unit_pos[self.player_idx][i]
            home_coords = self.get_home_coords()
            
            forces[unit].update({Tuple(home_coords): home_coords.distance(current_pos)})
            forces[unit].update(self.wall_forces(current_pos))
            forces[unit].update(self.closest_friend_force(i, current_pos, unit_pos, unit_id))
            forces[unit].update(danger_regions)

        return forces