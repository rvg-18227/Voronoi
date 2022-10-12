from collections import defaultdict
import os
import pickle
from matplotlib.pyplot import close
import numpy as np
import sympy
import logging
from typing import Tuple, List, Dict
import math
from shapely.geometry import Point, MultiPoint, LineString
from shapely.geometry.polygon import Polygon
from shapely.affinity import scale, rotate
from shapely.ops import nearest_points

from scipy.spatial.distance import cdist
import heapq

#HYPER PARAMETERS
DANGER_ZONE_RADIUS = 20

#SCISSOR STARTING OUTER RADIUS
OUTER_RADIUS = 50

#ANGLES FOR SCISSOR ZONE
SCISSOR_ZONE_COUNT = 5

#priority of regions
PRIORITY_ID = [4,2,0,1,3]

class ScissorRegion:
    def __init__(self, bounds, delta_bounds, id):

        #bounds are the two points defining the scissor ends of the lines
        self.bounds: Tuple[Tuple[float, float],Tuple[float, float]] = bounds
        self.center_point: Tuple[float, float] = ((bounds[0][0]+bounds[1][0])/2 , (bounds[0][1]+bounds[1][1])/2)
        self.dist: float = math.dist(bounds[0], bounds[1])

        #0 means go to bound[0]
        #1 means go to bound[1]
        self.direction = 0
        self.id = id

        self.polygon = Polygon([bounds[0], bounds[1], delta_bounds[1], delta_bounds[0]])

    def points_inbetween_points(self, p1, p2, amount):
        return zip(np.linspace(p1[0], p2[0], amount),np.linspace(p1[1], p2[1], amount))

    def __hash__(self):
        return hash(self.center_point)

    def __lt__(self, other):
        return self.id < other.id

    def __repr__(self):
        return(str(self.id))

def sentinel_transform_moves(moves):
    moves_new = {}

    for key in moves:
        moves_new[int(key)] = moves[key]
    
    return moves_new

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

def get_board_regions(region_number):
    region_size = 100/region_number
    index = 0
    regions_by_id = {}
    for row_step in range(region_number):
        right_top_corner = (0, row_step*region_size)
        for column_step in range(region_number):
            left_top_corner = (right_top_corner[0] + region_size, right_top_corner[1])
            right_bottom_corner = (right_top_corner[0], right_top_corner[1] + region_size)
            left_bottom_corner = (right_top_corner[0] + region_size, right_top_corner[1] + region_size)
            #print(right_top_corner, left_top_corner, right_bottom_corner, left_bottom_corner)
            regions_by_id[index] = Polygon([right_top_corner, right_bottom_corner, left_bottom_corner,
                                                left_top_corner])
            right_top_corner = left_top_corner
            index += 1
    return regions_by_id

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
        self.ally_units: Dict[str, Point] = {}
        self.ally_units_yesterday: Dict[str, Point] = {}
        self.ally_killed_unit_ids = []
        self.enemy_units: Dict[str, Point] = {}
        self.enemy_units_yesterday: Dict[str, Point] = {}
        self.enemy_killed_unit_ids = []

        self.far_radius = OUTER_RADIUS
        self.scissor_bounds = self.create_bounds(self.far_radius)
        self.regions = self.create_scissor_regions(self.scissor_bounds, self.create_bounds(self.far_radius-0.1))

        #key: u_id, val: region
        #u_id is otw to region
        self.unit_otw_region = {}

        #number of units otw to region
        self.regions_uid_otw = defaultdict(lambda : 0)

        # Platoon variables
        self.platoons = {1: {'unit_ids': [], 'target': None}} # {platoon_id: {unit_ids: [...], target: unit_id}}

        #dictionary of entire board broken up into regions
        self.entire_board_regions = get_board_regions(5)

    def get_home_coords(self):
        if self.player_idx == 0:
            return Point(0.5, 0.5)
        elif self.player_idx == 1:
            return Point(0.5, 99.5)
        elif self.player_idx == 2:
            return Point(99.5, 99.5)
        elif self.player_idx == 3:
            return Point(99.5, 0.5)

    def transform_move (self, dist_ang: Tuple[float, float]) -> Tuple[float, float]:
        dist, rad_ang = dist_ang
        return (dist, rad_ang - (math.pi/2 * self.player_idx))

    def point_move(self, p1, p2):
        dist = min(1, math.dist(p1,p2))
        angle = sympy.atan2(p2[1] - p1[1], p2[0] - p1[0])
        return (dist, angle)

    def shapely_point_move(self, p1, p2):
        return self.point_move((p1.x, p1.y), (p2.x, p2.y))
    
    def fixed_formation_moves(self, unit_ids, angles, move_size=1) -> Dict[float, Tuple[float, float]]:
        return {
            int(unit_id) : 
                self.transform_move((move_size, angles[idx % len(angles)]*(math.pi / 180)))
            for idx, unit_id
            in enumerate(unit_ids)
        }
    
    def platoon_moves(self, unit_ids)  -> Dict[float, Tuple[float, float]]:
        moves = {}

        # Draft units into platoons
        drafted_unit_ids = [uid for platoon in self.platoons.values() for uid in platoon['unit_ids']]
        undrafted_unit_ids = [uid for uid in unit_ids if uid not in drafted_unit_ids]
        for unit_id in undrafted_unit_ids:
            newest_platon_id = max(list(self.platoons.keys()))
            newest_platoon = self.platoons[newest_platon_id]
            if len(newest_platoon['unit_ids']) < 3:
                newest_platoon['unit_ids'].append(unit_id)
            else:
                self.platoons[newest_platon_id+1] = {'unit_ids': [unit_id], 'target': None}

        for platoon in self.platoons.values():
            # Remove killed units
            for uid in self.ally_killed_unit_ids:
                if uid in platoon['unit_ids']:
                    platoon['unit_ids'].remove(uid)
            
            # Remove target when it has been killed
            if platoon['target'] in self.enemy_killed_unit_ids:
                platoon['target'] = None

            # Assign targets for ready platoons
            if not platoon['target']  and len(platoon['unit_ids']) == 3:
                untargetted_enemy_unit_poss = [pos for uid, pos in self.enemy_units.items() if uid not in [p['target'] for p in self.platoons.values()]]
                enemy_unit_points = MultiPoint(untargetted_enemy_unit_poss)

                leader_point = self.ally_units[platoon['unit_ids'][0]]
                nearest_enemy_point = nearest_points(leader_point, enemy_unit_points)[1]
                platoon['target'] = list(self.enemy_units.keys())[list(self.enemy_units.values()).index(nearest_enemy_point)]
            
            # Generate moves for units in assigned platoons
            elif platoon['target']:
                platoon_unit_ids = platoon['unit_ids']
                for uid in platoon_unit_ids:
                    platoon_unit_idx = platoon_unit_ids.index(uid)
                    intersept_angle = self.intercept_angle(platoon['target'], uid, platoon_unit_idx)
                    if platoon_unit_idx == 0 and self.ally_units[platoon_unit_ids[1]] == self.get_home_coords():
                        moves[uid] = (0, intersept_angle)
                    else:
                        moves[uid] = (1, intersept_angle)

        return {int(uid): move for uid, move in moves.items()}

    def intercept_angle(self, target_unit_id, chaser_unit_id, platoon_unit_idx=None) -> float:
        target_pos = self.enemy_units[target_unit_id]
        target_prev_pos = self.enemy_units_yesterday[target_unit_id]
        target_vec = LineString([target_prev_pos, target_pos])
        scaled_target_vec = scale(target_vec, xfact=100, yfact=100, origin=target_prev_pos)

        chaser_pos = self.ally_units[chaser_unit_id]
        chaser_vec = LineString([chaser_pos, (chaser_pos.x+1, chaser_pos.y)])

        intersect_pos = target_pos
        if target_pos != target_prev_pos:
            best_dist = math.inf
            for angle in range(360):
                rotated_chaser_vec = rotate(chaser_vec, angle, origin=chaser_pos)
                scaled_chaser_vec = scale(rotated_chaser_vec, xfact=100, yfact=100, origin=chaser_pos)
                new_intersect_pos = scaled_chaser_vec.intersection(scaled_target_vec)
                intersect_dist = chaser_pos.distance(new_intersect_pos)
                if intersect_dist < best_dist and intersect_dist > 0:
                    intersect_pos = new_intersect_pos
                    best_dist = intersect_dist
        
        if target_pos != intersect_pos:
            chaser_to_intersect = LineString([chaser_pos, intersect_pos])
            left_flank = chaser_to_intersect.parallel_offset(4, 'left').boundary[1]
            right_flank = chaser_to_intersect.parallel_offset(4, 'right').boundary[0]

            if platoon_unit_idx == 1:
                intersect_pos = left_flank
            elif platoon_unit_idx == 2:
                intersect_pos = right_flank            
        
        return math.atan2(intersect_pos.y-chaser_pos.y, intersect_pos.x-chaser_pos.x)

    def sentinel_moves(self, unit_pos, unit_id) -> Dict[float, Tuple[float, float]]:

        moves = {}
        enemy_count = {}

        for r in self.regions:
            center_point = r.center_point
            enemy_count_in_region = self.enemy_count_in_region(unit_pos, center_point, self.player_idx, r.dist+4)
            enemy_count[r] = enemy_count_in_region

        region_contains_id, ids_in_region = self.regions_contain_id(unit_pos, unit_id)

        #Move to quadrant with (in priority, highest first):
        #no units > danger score > closest

        pqueue = []

        #create priority list
        for r in self.regions:
            
            units_commited = len(region_contains_id[r])+self.regions_uid_otw[r]
            
            score = float("inf")
            if len(region_contains_id[r]) > 0:
                score = enemy_count[r] / len(region_contains_id[r])

            element = (units_commited, score, r)

            heapq.heappush(pqueue, element)

        for i in range(len(unit_pos[self.player_idx])):
            pt = unit_pos[self.player_idx][i]
            curr = (pt.x, pt.y)
            u_id = unit_id[self.player_idx][i]

            #scissor motion
            if u_id in ids_in_region:

                #remove from count of region otw of r
                self.regions_uid_otw[r] -= 1

                #remove pathing to region
                self.unit_otw_region.pop(u_id, None)

                moves[u_id] = (0,0)
                #do scissoring stuff

            #if predefined from previous turn
            #move to point in region
            elif u_id in self.unit_otw_region:
                moves[u_id] = self.point_move(curr, self.unit_otw_region[u_id].center_point)

            else:
                #find the quadrant in need the most
                element = heapq.heappop(pqueue)
                dire_region = element[2]

                #send our u_id to a region
                self.unit_otw_region[u_id] = dire_region
                moves[u_id] = self.point_move(curr, self.unit_otw_region[u_id].center_point)

                #increment number of units otw to a region
                self.regions_uid_otw[dire_region] += 1

                #add back to the queue
                element_new = (element[0]+1, element[1], dire_region)

                heapq.heappush(pqueue, element_new)

        return sentinel_transform_moves(moves)

    def regions_contain_id(self, unit_pos, unit_id):

        team_set = defaultdict(lambda: set())

        all_units_in_a_region = set()

        for i in range(len(unit_pos[self.player_idx])):
            u_id = unit_id[self.player_idx][i]
            u_pt = unit_pos[self.player_idx][i]
            for r in self.regions:
                if r.polygon.contains(u_pt):
                    team_set[r].add(u_id)
                    all_units_in_a_region.add(u_id)
                    continue
        
        return team_set, all_units_in_a_region


    def create_bounds(self, radius_from_origin):

        scizzor_angle = np.linspace(0, 90, SCISSOR_ZONE_COUNT+1)

        #change angles to Radians
        #make them agnostic to player idx
        angles = [rad_ang*(math.pi / 180) - (math.pi/2 * self.player_idx) for rad_ang in scizzor_angle]

        bounds = []

        for a in angles:
            bounds.append((radius_from_origin*math.cos(a),radius_from_origin*math.sin(a)))

        return bounds

    def create_scissor_regions(self, bounds, delta_bounds):

        regions = []

        for i in range(len(bounds)-1):
            left_bound = bounds[i]
            right_bound = bounds[i+1]

            dl = delta_bounds[i]
            dr = delta_bounds[i+1]

            regions.append(ScissorRegion((left_bound, right_bound), (dl, dr), PRIORITY_ID[i]))

        return regions

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

        # Cache previous day's unit positions
        self.ally_units_yesterday = self.ally_units.copy()
        self.enemy_units_yesterday = self.enemy_units.copy()

        # Update current unit positions
        self.ally_units = {}
        self.enemy_units = {}
        for idx in range(4):
            if self.player_idx == idx:
                self.ally_units.update({uid: pos for uid, pos in zip(unit_id[idx], unit_pos[idx])})
            else:
                self.enemy_units.update({f"{idx}-{uid}": pos for uid, pos in zip(unit_id[idx], unit_pos[idx])})

        # Detect killed units
        self.ally_killed_unit_ids = [id for id in list(self.ally_units_yesterday.keys()) if id not in list(self.ally_units.keys())]
        self.enemy_killed_unit_ids = [id for id in list(self.enemy_units_yesterday.keys()) if id not in list(self.enemy_units.keys())]

        # Initialize all unit moves to null so that an unspecified move is equivalent to not moving
        moves = {int(id): (0, 0) for id in unit_id[self.player_idx]}

        if self.player_idx == 0:
            # Max takes 0
            moves.update(self.sentinel_moves(unit_pos, unit_id))
        elif self.player_idx == 1:
            self.get_forces(unit_id, unit_pos)
            # Abigail takes 1
            moves.update(self.fixed_formation_moves(unit_id[self.player_idx][:1], [45.0], 0 if unit_pos[self.player_idx][0].x >= 25 else 1)) 
        elif self.player_idx == 2:
            # Noah takes 2
            moves.update(self.platoon_moves(unit_id[self.player_idx]))

        elif self.player_idx == 3:
            pass

        self.days += 1  
        return list(moves.values())

    #what is the enemy_count team_count for a given point
    def enemy_count_in_region(self, unit_pos, curr_point_coords, team_idx, radius):
        np_unit_pos = points_to_numpy(unit_pos)
        distances = cdist([np.array((curr_point_coords[0], curr_point_coords[1]))], np_unit_pos[:, :2]).flatten()
        close_points = np_unit_pos[distances <= radius,:]
        team_points = close_points[close_points[:,2] == team_idx]

        close_team_count = team_points.shape[0]
        close_enemy_count = close_points.shape[0]-close_team_count

        return close_enemy_count

    #for a given unit, given by u
    def danger_score_of_point(self, np_unit_pos, u, team_idx, radius) -> float:
        distances = cdist([np.array((u.x, u.y))], np_unit_pos[:, :2]).flatten()
        close_points = np_unit_pos[distances <= radius,:]
        team_points = close_points[close_points[:,2] == team_idx]

        #should always be >=1 because it counts itself
        close_team_count = team_points.shape[0]

        close_enemy_count = close_points.shape[0]-close_team_count

        danger_score = close_enemy_count/close_team_count
        
        return danger_score

    #returns a dict of dicts, for the danger regions of all u_id
    #dict key is team_idx
    #dict val is dict {u_id : danger_score}
    def danger_levels(self, unit_pos, unit_id) -> Dict[float, Dict[float, float]]:
        np_unit_pos = points_to_numpy(unit_pos)
        danger_score = [{} for _ in range(4)]

        for team_idx in range(4):
            for idx, u in enumerate(unit_pos[team_idx]):

                u_id = int(unit_id[team_idx][idx])

                danger_score[team_idx][u_id] = self.danger_score_of_point(np_unit_pos, u, team_idx, DANGER_ZONE_RADIUS)

        #result is the danger level of all units (same dimensions as unit_pos)

        #danger_regions is a dict with {(center_point of region) : sum of danger_level of teammate}

        #region_count is number of teammates in region

        #unit_regions is which region is a unit in
        return danger_score

    def wall_forces(self, current_point) -> List[Tuple[Tuple[float, float], float]]:
        current_x = current_point.x
        current_y = current_point.y
        dist_to_top = Point(current_x, 0).distance(current_point)
        dist_to_bottom = Point(current_x, 100).distance(current_point)
        dist_to_right = Point(0, current_y).distance(current_point)
        dist_to_left = Point(100, current_y).distance(current_point)
        return [((current_x, 0), dist_to_top), ((current_x, 100), dist_to_bottom), ((100, current_y), dist_to_right), ((0, current_y), dist_to_left)]

    def closest_friend_force(self, current_unit, current_pos, unit_pos, unit_id) -> List[Tuple[Tuple[float, float], float]]:
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
        return [((friend_unit_pos.x, friend_unit_pos.y), closest_unit_dist)]

    def least_popular_region_force(self, unit_pos):
        number_regions = len(self.entire_board_regions)
        unit_per_region = np.zeros(number_regions)
        for index in range(number_regions):
            current_poly = self.entire_board_regions[index]
            for player_num in range(4):
                for unit in unit_pos[player_num]:
                    if current_poly.contains(unit):
                        unit_per_region[index] += 1
        index_min_region = int(np.argmin(unit_per_region))
        min_poly = self.entire_board_regions[index_min_region]
        center = min_poly.centroid
        #print(center)
        return [(index_min_region, (center.x, center.y))]


    def get_forces(self, unit_id, unit_pos):# danger_regions):
        forces = {id: [] for id in unit_id[self.player_idx]}
        for i in range(len(unit_id[self.player_idx])):
            unit = unit_id[self.player_idx][i]
            current_pos = unit_pos[self.player_idx][i]
            home_coords = self.get_home_coords()
            
            forces[unit].append([(home_coords.x, home_coords.y), home_coords.distance(current_pos)])
            forces[unit].append(self.wall_forces(current_pos))
            forces[unit].append(self.closest_friend_force(i, current_pos, unit_pos, unit_id))
            forces[unit].append(self.least_popular_region_force(unit_pos))
        return forces