from collections import defaultdict
import os
import pickle
from turtle import home
from matplotlib import units
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

DETECTION_RADIUS = 15
DELTA_RADIUS = 2

#SCISSOR STARTING OUTER RADIUS
OUTER_RADIUS = 55
INNER_RADIUS = OUTER_RADIUS-3

OUTER_SPEED = 1
INNER_SPEED = 1

#ANGLES FOR SCISSOR ZONE
SCISSOR_ZONE_COUNT = 3
REGION_INCREMENT = 0.5

#priority of regions
if SCISSOR_ZONE_COUNT == 5:
    PRIORITY_ID_OUTER = [4,2,0,1,3]
    PRIORITY_ID_INNER = [9,7,5,6,8]
elif SCISSOR_ZONE_COUNT == 6:
    PRIORITY_ID_OUTER = [4,2,0,1,3,5]
    PRIORITY_ID_INNER = [11,9,7,8,10,12]
elif SCISSOR_ZONE_COUNT == 3:
    #priority of regions for 3 zones
    PRIORITY_ID_OUTER = [2,0,1]
    PRIORITY_ID_INNER = [7,5,6]

#fixed formation count
FIXED_FORMATION_COUNT = 3

class ScissorRegion:
    def __init__(self, bounds, delta_bounds, detection_bounds, id, player_idx, angles, radius, speed):

        #bounds are the two points defining the scissor ends of the lines
        self.detection_bounds: Tuple[Tuple[float, float],Tuple[float, float]] = detection_bounds
        self.bounds: Tuple[Tuple[float, float],Tuple[float, float]] = bounds
        self.delta_bounds: Tuple[Tuple[float, float],Tuple[float, float]] = delta_bounds
        self.center_point: Tuple[float, float] = self.find_cp((self.find_cp(bounds), self.find_cp(delta_bounds)))
        self.angles: List[float, float] = angles

        #0 means go to bound[0]
        #1 means go to bound[1]
        self.direction = (id)%2
        self.target_point = self.bounds[self.direction]
        self.id = id
        self.player_idx = player_idx

        self.radius = radius

        self.polygon = Polygon([bounds[0], bounds[1], delta_bounds[1], delta_bounds[0]])
        self.detection_polygon = Polygon([get_corner_coords(self.player_idx), detection_bounds[1], detection_bounds[0]])

        self.connected_scissor_region : ScissorRegion = None
        self.speed = speed

        #for units in region
        self.targets = {}

    #call everyturn
    def update_targets(self, units, unit_pos):

        self.target_point = self.bounds[self.direction]

        #assume units are sorted here
        #units in u_id are in the region
        #units is a list of [u_id], where the it is sorted by distance to self.target_point
        n = len(units)

        sorting_list = []

        target = Point(self.target_point[0], self.target_point[1]) 

        for u_id in units:
            sorting_list.append((unit_pos[u_id].distance(target),u_id))

        sorting_list.sort()
        sorted_units =  [x[1] for x in sorting_list]

        target = self.target_point
        opposite = self.bounds[1]

        if self.direction == 1:
            opposite = self.bounds[0]

        density = max((self.radius-OUTER_RADIUS)//10, 2)

        target_x = np.linspace(target[0], opposite[0], n+2)
        target_y = np.linspace(target[1], opposite[1], n+2)

        result = {}

        for i in range(len(units)):
            result[sorted_units[i]] = (target_x[i], target_y[i])

        self.targets = result
        return result


    def update_polygons(self):
        self.polygon = Polygon([self.bounds[0], self.bounds[1], self.delta_bounds[1], self.delta_bounds[0]])
        self.detection_polygon = Polygon([get_corner_coords(self.player_idx), self.detection_bounds[1], self.detection_bounds[0]])

    def find_cp(self, bounds):
        return ((bounds[0][0]+bounds[1][0])/2 , (bounds[0][1]+bounds[1][1])/2)

    def changeDirection(self):
        #print("DIRECTION CHANGED")
        if self.direction == 0:
            self.direction = 1
        else:
            self.direction = 0

        #if self.connected_scissor_region is not None:
        #    self.connected_scissor_region.changeDirectionHelper(self.direction)

    """def changeDirectionHelper(self, sister_direction):
        if sister_direction == 0:
            self.direction = 1
        else:
            self.direction = 0

        self.target_point = self.bounds[self.direction]"""

    def changeBounds(self, radius_increment):

        increment_l = find_increment(radius_increment, self.angles[0])
        increment_r = find_increment(radius_increment, self.angles[1])

        self.radius += radius_increment

        self.bounds = increment_bounds(self.bounds, increment_l, increment_r)
        self.delta_bounds = increment_bounds(self.delta_bounds, increment_l, increment_r)
        self.detection_bounds = increment_bounds(self.detection_bounds, increment_l, increment_r)

        self.center_point: Tuple[float, float] = self.find_cp((self.find_cp(self.bounds), self.find_cp(self.delta_bounds)))
        self.update_polygons()

        if self.connected_scissor_region is not None:
            self.connected_scissor_region.changeBoundsHelper(radius_increment)
                
    def changeBoundsHelper(self, radius_increment):

        increment_l = find_increment(radius_increment, self.angles[0])
        increment_r = find_increment(radius_increment, self.angles[1])

        self.radius += radius_increment

        self.bounds = increment_bounds(self.bounds, increment_l, increment_r)
        self.delta_bounds = increment_bounds(self.delta_bounds, increment_l, increment_r)
        self.detection_bounds = increment_bounds(self.detection_bounds, increment_l, increment_r)

        self.center_point: Tuple[float, float] = self.find_cp((self.find_cp(self.bounds), self.find_cp(self.delta_bounds)))
        self.update_polygons()

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return str(self.id)

def find_increment(new_radius, a):
    new_point = (new_radius*math.cos(a),new_radius*math.sin(a))
    return new_point

def increment_bounds(bound, incrementl, incrementr):
    return ((bound[0][0]+incrementl[0], bound[0][1]+incrementl[1]),(bound[1][0]+incrementr[0], bound[1][1]+incrementr[1]))

def get_corner_coords(team_idx):
    if team_idx == 0:
        return Point(0, 0)
    elif team_idx == 1:
        return Point(0, 100)
    elif team_idx == 2:
        return Point(100, 100)
    elif team_idx == 3:
        return Point(100, 0)

def create_bounds(radius_from_origin, team_idx):

    scizzor_angle = np.linspace(0, 90, SCISSOR_ZONE_COUNT+1)

    home = get_corner_coords(team_idx)
    #change angles to Radians
    #make them agnostic to player idx
    angles = [rad_ang*(math.pi / 180) - (math.pi/2 * team_idx) for rad_ang in scizzor_angle]

    bounds = []

    for a in angles:
        bounds.append((radius_from_origin*math.cos(a)+home.x,radius_from_origin*math.sin(a)+home.y))

    return bounds, angles

def create_scissor_regions(radius, team_idx, prio_idx, speed):

    bounds, a = create_bounds(radius, team_idx)
    delta_bounds, a = create_bounds(radius-DELTA_RADIUS, team_idx)
    detection_bounds, a = create_bounds(radius+DETECTION_RADIUS, team_idx)

    regions = []

    for i in range(len(bounds)-1):
        left_bound = bounds[i]
        right_bound = bounds[i+1]

        dl = delta_bounds[i]
        dr = delta_bounds[i+1]

        dtct_l = detection_bounds[i]
        dtct_r = detection_bounds[i+1]

        regions.append(ScissorRegion((left_bound, right_bound), (dl, dr), (dtct_l, dtct_r), prio_idx[i], team_idx, [a[i], a[i+1]], radius, speed))

    return regions


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


def get_regions_away_home(region_num, home):
    region_dict = get_board_regions(region_num)
    region_copy = region_dict.copy()
    home_coord = home
    for unit_id in region_dict:
        region = region_dict[unit_id]
        current_center = region.centroid
        if home_coord.distance(current_center) < INNER_RADIUS:
            del region_copy[unit_id]
    return region_copy

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

        #game play
        self.spawn_days = spawn_days
        self.game_length = total_days

        self.rng = rng
        self.logger = logger
        self.player_idx = player_idx
        self.days = 1
        self.ally_units: Dict[str, Point] = {}
        self.ally_unit_tuples: Dict[str, Tuple[float, float]] = {}
        self.ally_units_yesterday: Dict[str, Point] = {}
        self.ally_killed_unit_ids = []
        self.historical_ally_unit_ids = set() 
        self.enemy_units: Dict[str, Point] = {}
        self.enemy_unit_tuples: Dict[str, Tuple[float, float]] = {}
        self.enemy_units_yesterday: Dict[str, Point] = {}
        self.enemy_killed_unit_ids = []
        self.home_coords = self.get_home_coords()
        self.home_coord_tuple = (self.home_coords.x, self.home_coords.y)
        self.all_unit_pos_tuples = []

        self.regions = create_scissor_regions(OUTER_RADIUS, player_idx, PRIORITY_ID_OUTER, OUTER_SPEED)
        self.regions += create_scissor_regions(INNER_RADIUS, player_idx, PRIORITY_ID_INNER, INNER_SPEED)

        for idx in range(SCISSOR_ZONE_COUNT):
            self.regions[idx].connected_scissor_region = self.regions[idx+SCISSOR_ZONE_COUNT]
            self.regions[idx+SCISSOR_ZONE_COUNT].connected_scissor_region = self.regions[idx]

        #key: u_id, val: region
        #u_id is otw to region
        self.unit_otw_region = {}
        self.unit_commited_region = {}

        #number of units otw to region
        self.regions_uid_otw = defaultdict(lambda : set())
        self.regions_uid_commited = defaultdict(lambda: set())

        #memoize the previous enemies
        self.enemy_in_region = {}

        # Platoon variables
        self.platoons = {1: {'unit_ids': [], 'target': None}} # {platoon_id: {unit_ids: [...], target: unit_id}}
        self.defender_platoon_ids = []
        self.num_defender_platoons = 3
        self.attacker_platoon_ids = []
        self.platoon_ids_waiting_for_replenishment_units = set()

        #dictionary of entire board broken up into regions
        self.entire_board_regions = get_regions_away_home(5, self.home_coords)
        self.entire_board_region_centroids = {id: region.centroid for id, region in self.entire_board_regions.items()}
        self.entire_board_region_tuples = {idx: region.bounds for idx, region in self.entire_board_regions.items()}
        #dict of scouts and their current region id occupied
        self.scout = dict.fromkeys(region_id for region_id in self.entire_board_regions)

    def get_home_coords(self):
        if self.player_idx == 0:
            return Point(0.5, 0.5)
        elif self.player_idx == 1:
            return Point(0.5, 99.5)
        elif self.player_idx == 2:
            return Point(99.5, 99.5)
        else:
            return Point(99.5, 0.5)

    def transform_move(self, dist_ang: Tuple[float, float]) -> Tuple[float, float]:
        dist, rad_ang = dist_ang
        return (dist, rad_ang - (math.pi/2 * self.player_idx))

    def transform_angle(self, ang) -> float:
        return ang - (math.pi/2 * self.player_idx)

    def point_move(self, p1, p2):
        dist = min(1, math.dist(p1,p2))
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        return (dist, angle)
    
    def point_move_within_scissor(self, p1, p2, max_dist=1):
        dist = min(max_dist, math.dist(p1,p2)-0.01)
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
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

    def short_game_moves(self, unit_ids, move_size=1):
        number_units = self.game_length/self.spawn_days
        angle_jump = 90/number_units
        unit_id_list = []
        angle_list = []
        for unit_id in unit_ids:
            unit_id = int(unit_id)
            if unit_id == 1:
                unit_id_list.append(unit_id)
                angle_increment = 45
                angle_list.append(angle_increment)
            elif unit_id == 2:
                unit_id_list.append(unit_id)
                angle_increment = 80
                angle_list.append(angle_increment)
            elif unit_id == 3:
                unit_id_list.append(unit_id)
                angle_increment = 10
                angle_list.append(angle_increment)
            elif unit_id == 4:
                unit_id_list.append(unit_id)
                angle_increment = 62
                angle_list.append(angle_increment)
            elif unit_id == 5:
                unit_id_list.append(unit_id)
                angle_increment = 27
                angle_list.append(angle_increment)
            elif int(unit_id) % 2 == 0:
                unit_id_list.append(unit_id)
                angle_increment = (unit_id -5 )*angle_jump % 90
                angle_list.append(angle_increment)
            else:
                unit_id_list.append(unit_id)
                angle_increment = (90 - (unit_id -5) *angle_jump) % 90
                angle_list.append(angle_increment)

        return self.fixed_formation_moves(unit_id_list, angle_list)
    
    def clamp_point_within_map(self, point: Point) -> Point:
        x = min(99.99, max(0.01, point.x))
        y = min(99.99, max(0.01, point.y))
        return Point(x, y)

    def platoon_unit_moves(self, platoon_id, intercept_pos, overshoot=True) -> Dict[float, Tuple[float, float]]:
        moves = {}
        unit_ids = self.platoons[platoon_id]['unit_ids']
        leader_pos = self.ally_units[unit_ids[0]]
        left_flank_pos = self.ally_units[unit_ids[1]]
        right_flank_pos = self.ally_units[unit_ids[2]] 

        # Generate moves based on path from leader to 1m past intercept point
        chaser_to_intersect = LineString([leader_pos, intercept_pos])
        chaser_to_intersect_dist = chaser_to_intersect.length

        if not overshoot and chaser_to_intersect_dist <= 1:
            return moves

        scale_factor = (chaser_to_intersect_dist + 2) / chaser_to_intersect_dist
        scaled_chaser_to_intersect = scale(chaser_to_intersect, xfact=scale_factor, yfact=scale_factor, origin=leader_pos)
        leader_dest_pos = scaled_chaser_to_intersect.interpolate(1)

        # Flanks should aim for points past the leader's destination, to the left and right
        leader_pos_to_flank_center = LineString([leader_pos, scaled_chaser_to_intersect.interpolate(3.5)])
        if leader_dest_pos != intercept_pos:
            left_flank_dest_pos = leader_pos_to_flank_center.parallel_offset(1, 'left').boundary.geoms[1]
            right_flank_dest_pos = leader_pos_to_flank_center.parallel_offset(1, 'right').boundary.geoms[0]
        else:
            left_flank_dest_pos = leader_dest_pos
            right_flank_dest_pos = leader_dest_pos

        # Ensure that all moves stay within the map's bounds
        leader_dest_pos = self.clamp_point_within_map(leader_dest_pos)
        left_flank_dest_pos = self.clamp_point_within_map(left_flank_dest_pos)
        right_flank_dest_pos = self.clamp_point_within_map(right_flank_dest_pos)

        # The leader will wait for the flanks to get in position before moving out
        if left_flank_pos.distance(left_flank_dest_pos) > 1.2 or right_flank_pos.distance(right_flank_dest_pos) > 1.2:
            leader_dest_pos = leader_pos

        for idx, uid in enumerate(unit_ids):
            if idx == 0:
                moves[uid] = (leader_pos.distance(leader_dest_pos), math.atan2(leader_dest_pos.y-leader_pos.y, leader_dest_pos.x-leader_pos.x))
            elif idx == 1:
                moves[uid] = (left_flank_pos.distance(left_flank_dest_pos), math.atan2(left_flank_dest_pos.y-left_flank_pos.y, left_flank_dest_pos.x-left_flank_pos.x))
            elif idx == 2:
                moves[uid] = (right_flank_pos.distance(right_flank_dest_pos), math.atan2(right_flank_dest_pos.y-right_flank_pos.y, right_flank_dest_pos.x-right_flank_pos.x))
                            
        return moves
    
    def platoon_moves(self, unit_ids)  -> Dict[float, Tuple[float, float]]:
        moves = {}
        home_point = self.home_coords

        # Draft units into platoons
        drafted_unit_ids = [uid for platoon in self.platoons.values() for uid in platoon['unit_ids']]
        undrafted_unit_ids = [uid for uid in unit_ids if uid not in drafted_unit_ids]
        for unit_id in undrafted_unit_ids:
            # Replenish platoons that have lost units first
            not_full_platoons = [(pid, p) for pid, p in self.platoons.items() if len(p['unit_ids'])<3]
            if len(not_full_platoons) >= 1:
                not_full_platoons[0][1]['unit_ids'].append(unit_id)
                self.platoon_ids_waiting_for_replenishment_units.add(not_full_platoons[0][0])
            else:
                # Create a new platoon if all existing platoons are full
                newest_platon_id = max(list(self.platoons.keys()))
                self.platoons[newest_platon_id+1] = {'unit_ids': [unit_id], 'target': None}

        # Remove killed platoons
        platoon_ids_to_remove = []
        for pid, platoon in self.platoons.items():
            if len(platoon['unit_ids']) == 0 and pid != 1:
                platoon_ids_to_remove.append(pid)
        for pid in platoon_ids_to_remove:
            del self.platoons[pid]
            if pid in self.defender_platoon_ids:
                self.defender_platoon_ids.remove(pid)
            if pid in self.attacker_platoon_ids:
                self.attacker_platoon_ids.remove(pid)
            if pid in self.platoon_ids_waiting_for_replenishment_units:
                self.platoon_ids_waiting_for_replenishment_units.remove(pid)

        for pid, platoon in self.platoons.items():
            # Remove killed units
            for uid in self.ally_killed_unit_ids:
                if uid in platoon['unit_ids']:
                    platoon['unit_ids'].remove(uid)

            # Remove target when it has been killed, a new target will be assigned
            if platoon['target'] in self.enemy_killed_unit_ids:
                platoon['target'] = None

            # Assign targets for ready platoons
            if not platoon['target'] and len(platoon['unit_ids']) == 3:
                min_dist = math.inf
                target_id = None

                enemy_unit_ids_in_territory = []
                enemy_unit_ids_encroaching_on_territory = []
                for uid, pos in self.enemy_units.items():
                    dist_to_home = home_point.distance(pos)
                    if dist_to_home <= INNER_RADIUS:
                        enemy_unit_ids_in_territory.append((uid, pos))
                    elif dist_to_home >= OUTER_RADIUS and dist_to_home <= (OUTER_RADIUS + 20):
                        enemy_unit_ids_encroaching_on_territory.append((uid, pos))

                if pid not in self.defender_platoon_ids and pid not in self.attacker_platoon_ids:
                    if len(self.defender_platoon_ids) < 3:
                        self.defender_platoon_ids.append(pid)
                    else:
                        self.attacker_platoon_ids.append(pid)

                targetable_units = enemy_unit_ids_in_territory if pid in self.defender_platoon_ids else enemy_unit_ids_encroaching_on_territory 

                leader_point = self.ally_units[platoon['unit_ids'][0]]
                for uid, pos in targetable_units:
                    dist = leader_point.distance(pos)
                    if dist < min_dist:
                        min_dist = dist
                        target_id = uid

                # Send waiting defender platoons to waiting positions
                if target_id == None and pid in self.defender_platoon_ids:
                    home_to_waiting_point = LineString([home_point, Point(50, 50)])
                    angles = np.linspace(-35, 35, self.num_defender_platoons)
                    rot_angle = self.transform_angle(angles[self.defender_platoon_ids.index(pid)])
                    home_to_waiting_point = rotate(home_to_waiting_point, rot_angle, home_point)
                    waiting_point = home_to_waiting_point.interpolate(INNER_RADIUS / 2)
                    moves.update(self.platoon_unit_moves(pid, waiting_point, False))      
                
                platoon['target'] = target_id
            
            # Generate moves for units in assigned platoons
            if platoon['target'] and len(platoon['unit_ids']) == 3:
                intercept_point = self.intercept_point(platoon['target'], platoon['unit_ids'][0])
                moves.update(self.platoon_unit_moves(pid, intercept_point))

            # Send assigned platoons with lost units back towards home to meet their replenishments
            elif (pid in self.platoon_ids_waiting_for_replenishment_units and len(platoon['unit_ids']) >=1):
                point_distances = [self.ally_units[uid].distance(home_point) for uid in platoon['unit_ids']]
                furthest_dist = max(point_distances)
                if furthest_dist != 0:
                    furtherst_point_idx = point_distances.index(furthest_dist)
                    furtherst_point = self.ally_units[platoon['unit_ids'][furtherst_point_idx]]
                    home_to_furthest = LineString([home_point, furtherst_point])
                    meeting_point = home_to_furthest.interpolate(home_to_furthest.length * (0.5 if pid in self.defender_platoon_ids else 0.8))
                    for uid in platoon['unit_ids']:
                        moves[uid] = self.shapely_point_move(self.ally_units[uid], meeting_point)
            
                # Allow platoons awaiting replenishments to resume normal targetting when sufficiently close
                if pid in self.platoon_ids_waiting_for_replenishment_units and len(platoon['unit_ids']) == 3:
                    leader_point = self.ally_units[platoon['unit_ids'][0]]
                    lf_point = self.ally_units[platoon['unit_ids'][1]]
                    rf_point = self.ally_units[platoon['unit_ids'][0]]
                    if leader_point.distance(lf_point) < 3 and leader_point.distance(rf_point) < 3:
                        self.platoon_ids_waiting_for_replenishment_units.remove(pid)
        
        return {int(uid): move for uid, move in moves.items()}

    def scout_moves(self, unit_ids, map_states) -> Dict[float, Tuple[float, float]]:
        moves = {}
        unit_forces = self.get_forces(unit_ids, map_states)

        current_scout_ids = [scout_id for scout_id in self.scout.values()]
        free_unit_ids = [uid for uid in unit_ids if uid not in current_scout_ids]

        for region in self.scout:
            # Remove killed units
            for uid in self.ally_killed_unit_ids:
                if uid == self.scout[region]:
                    self.scout[region] = None

        min_home = math.inf
        min_unit_id = math.inf
        for unit_id in free_unit_ids:
            current_dist_home = unit_forces[unit_id][0][1]
            if current_dist_home < min_home:
                min_home = current_dist_home
                min_unit_id = unit_id
                if int(min_home) == 0:
                    break
        if min_home != math.inf:
            #print("calculating")
            least_pop_region_id = self.least_popular_region_force(map_states)
            least_pop_region_id = least_pop_region_id[0][0]
            if least_pop_region_id is not None:
                    #print(least_pop_region_id)
                self.scout[least_pop_region_id] = min_unit_id


        # Assign movement path towards empty region for scout
        for region in self.scout:
            move_dist = None
            angle = None
            if self.scout[region] is not None:
                current_scout_pos = self.ally_units[self.scout[region]]
                region_center_point = self.entire_board_region_centroids[region] #unit_forces[self.scout[region]][3][0][1]
                region_center_point = (region_center_point.x, region_center_point.y)
                if current_scout_pos.distance(Point(region_center_point)) == 0:
                    move_dist = 0
                    angle = 0
                else:
                    move_dist = 1
                    angle = math.atan2(region_center_point[1] - current_scout_pos.y,
                                        region_center_point[0] - current_scout_pos.x)

                moves[self.scout[region]] = (move_dist, angle)
        return {int(uid): move for uid, move in moves.items()}

    def intercept_point(self, target_unit_id, chaser_unit_id) -> float:
        target_pos = self.enemy_units[target_unit_id]
        target_prev_pos = self.enemy_units_yesterday[target_unit_id]
        target_vec = LineString([target_prev_pos, target_pos])
        scaled_target_vec = scale(target_vec, xfact=100, yfact=100, origin=target_prev_pos)

        chaser_pos = self.ally_units[chaser_unit_id]
        chaser_vec = LineString([chaser_pos, (chaser_pos.x+1, chaser_pos.y)])
        scaled_chaser_vec = scale(chaser_vec, xfact=100, yfact=100, origin=chaser_pos)

        chaser_to_target_ang = round(math.atan2(chaser_pos.y-chaser_pos.y, target_pos.x-target_pos.x) * (180/math.pi))
        # Search for best intercept point by testing a variety of chaser angles    
        intersect_pos = target_pos
        if target_pos != target_prev_pos:
            best_dist = math.inf
            for angle in range(chaser_to_target_ang-45, chaser_to_target_ang+45, 2):
                rotated_chaser_vec = rotate(scaled_chaser_vec, angle, origin=chaser_pos)
                new_intersect_pos = rotated_chaser_vec.intersection(scaled_target_vec)
                chaser_to_intersect_dist = chaser_pos.distance(new_intersect_pos)
                target_to_intersect_dist = target_pos.distance(new_intersect_pos)

                if (chaser_to_intersect_dist + target_to_intersect_dist) < best_dist and chaser_to_intersect_dist > 0:
                    intersect_pos = new_intersect_pos
                    best_dist = chaser_to_intersect_dist + target_to_intersect_dist

            # If our target distance to intercept is much larger than the chaser distance to intercept,
            # they are likely coming right for us, so we should aim for halfway between chaser and target
            if target_pos.distance(intersect_pos) / chaser_pos.distance(intersect_pos) >= 10:
                chaser_to_target = LineString([chaser_pos, target_pos])
                intersect_pos = chaser_to_target.interpolate(chaser_pos.distance(target_pos)/2)
        
        return intersect_pos

    def sort_by_distance(self, unit_id, r):

        sorting_list = []

        target = Point(r.target_point[0], r.target_point[1]) 

        for u_id in unit_id:
            sorting_list.append((self.ally_units[u_id].distance(target),u_id))

        sorting_list.sort()
        return [x[1] for x in sorting_list]

    def sentinel_moves(self, unit_ids) -> Dict[float, Tuple[float, float]]:

        moves = {}
        enemy_count = self.enemy_count_in_region()

        region_contains_id, uid_in_region = self.regions_contain_id(unit_ids)

        #move units in regions
        #scissor motion
        #if unit is in a point

        #check if which regions we need to change direction
        for r in region_contains_id:
            target = Point(r.target_point[0], r.target_point[1]) 

            for u_id in region_contains_id[r]:
                if self.ally_units[u_id].distance(target) <= 0.1:
                    r.changeDirection()
                    break
                

        #check which regions we need to increase bounds, and handle movement within region
        for region in self.regions:

            #bound increments only dependent on outer region
            if region.id in PRIORITY_ID_OUTER:
                
                #units in region
                unit_count_in_region = 0
                if region in region_contains_id:
                    unit_count_in_region = len(region_contains_id[region])

                #units in inner region
                unit_count_in_inner_region = 0
                if region.connected_scissor_region in region_contains_id:
                    unit_count_in_inner_region = len(region_contains_id[region.connected_scissor_region])

                #total units in chunk
                ally_count = unit_count_in_region+unit_count_in_inner_region

                ally_net_players_inc = ally_count + len(self.regions_uid_otw[region]) - enemy_count[region]
                ally_net_players_dec = (ally_count)*11 - enemy_count[region]

                if (ally_net_players_inc >= 3 and unit_count_in_region >= 2) and region.radius < 70:

                    #move region up
                    region.changeBounds(REGION_INCREMENT)
                    #print("REGION {} INCREMENTING!".format(region.id))

                elif ally_net_players_dec < 0:

                    #move region back
                    region.changeBounds(-REGION_INCREMENT)


        #need to apply moves after figuring out new region direction and bounds
        for region in region_contains_id:

            units = region_contains_id[region]

            #update targetting for each unit
            region.update_targets(list(units), self.ally_units)

        #Move to quadrant with (in priority, highest first):
        #no units > danger score > closest
        pqueue = []

        #create priority list
        for r in self.regions:
            
            val = len(region_contains_id[r]) if r in region_contains_id else 0
            units_commited = val+len(self.regions_uid_otw[r])
            
            score = -float("inf")
            if val > 0:
                score = -enemy_count[r] / len(region_contains_id[r])

            element = (units_commited, score, r)

            heapq.heappush(pqueue, element)

        #units not in a region yet
        for u_id, pt in [(u_id, pt) for u_id, pt in self.ally_units.items() if u_id in unit_ids]:
            curr = (pt.x, pt.y)

            #region moved, and move has been computed already
            if u_id in uid_in_region:

                #target point
                target = uid_in_region[u_id].targets[u_id]

                moves[u_id] = self.point_move_within_scissor(curr, target, 1)

            #if a unit is already commited to a region
            elif u_id in self.unit_commited_region:
                moves[u_id] = self.point_move(curr, self.unit_commited_region[u_id].center_point)

            else:
                #find the quadrant in need the most
                element = heapq.heappop(pqueue)
                dire_region = element[2]

                #send our u_id to a region
                self.unit_otw_region[u_id] = dire_region
                self.unit_commited_region[u_id] = dire_region
                moves[u_id] = self.point_move(curr, self.unit_otw_region[u_id].center_point)

                #increment number of units otw to a region
                self.regions_uid_otw[dire_region].add(u_id)
                self.regions_uid_commited[dire_region].add(u_id)

                #add back to the queue
                element_new = (element[0]+1, element[1], dire_region)

                heapq.heappush(pqueue, element_new)

        #print(moves)
        return sentinel_transform_moves(moves)

    def regions_contain_id(self, unit_ids):

        team_set = {}
        uid_in_region = {}

        for u_id, u_pt in [(u_id, pt) for u_id, pt in self.ally_units.items() if u_id in unit_ids]:
            if u_id in self.unit_commited_region:

                r = self.unit_commited_region[u_id]

                if r.polygon.contains(u_pt):
                    if r not in team_set:
                        team_set[r] = set()
                    team_set[r].add(u_id)

                    uid_in_region[u_id] = r

                    self.unit_otw_region.pop(u_id, None)
                    self.regions_uid_otw[r].discard(u_id)

        return team_set, uid_in_region

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
        
        self.ally_unit_tuples = {uid: (pt.x, pt.y) for uid, pt in self.ally_units.items()}
        self.enemy_unit_tuples = {uid: (pt.x, pt.y) for uid, pt in self.enemy_units.items()}
        self.all_unit_pos_tuples = [pos for pos in list(self.ally_unit_tuples.values()) + list(self.enemy_unit_tuples.values())]

        # Detect killed units
        self.ally_killed_unit_ids = [id for id in list(self.ally_units_yesterday.keys()) if id not in list(self.ally_units.keys())]
        self.enemy_killed_unit_ids = [id for id in list(self.enemy_units_yesterday.keys()) if id not in list(self.enemy_units.keys())]

        # Keep a record of all unit_ids that ever existed
        self.historical_ally_unit_ids.update(list(self.ally_units.keys()))

        # Initialize all unit moves to null so that an unspecified move is equivalent to not moving
        moves = {int(id): (0, 0) for id in unit_id[self.player_idx]}
  
        alive_ally_unit_ids = list(self.ally_units.keys())
        assignable_ally_unit_ids = sorted(list(self.historical_ally_unit_ids), key=int)

        # if game length < 50, special plan
        if self.game_length <= 50:
            #print(assignable_ally_unit_ids)
            moves.update(self.short_game_moves(assignable_ally_unit_ids))
            return list(moves.values())

        if self.game_length/self.spawn_days < 25:
            moves = []
            angle_jump = 10
            angle_start = 45
            for i in range(len(unit_id[self.player_idx])):
                distance = 1

                angle = (((i) * (angle_jump) + angle_start ))%90

                moves.append((distance, angle* (math.pi / 180)))

            return [self.transform_move(move) for move in moves]

        #seperate fixed formation vs strategic allys
        fixed_formation_ally_unit_ids = assignable_ally_unit_ids[:FIXED_FORMATION_COUNT]
        assignable_ally_unit_ids = assignable_ally_unit_ids[FIXED_FORMATION_COUNT:]

        assignable_ally_unit_id_chunks = [assignable_ally_unit_ids[i:i+10] for i in range(0, len(assignable_ally_unit_ids), 10)]

        sentinel_unit_ids = \
            [chunk[0] for chunk in assignable_ally_unit_id_chunks if len(chunk) >= 1] + \
            [chunk[1] for chunk in assignable_ally_unit_id_chunks if len(chunk) >= 2] + \
            [chunk[2] for chunk in assignable_ally_unit_id_chunks if len(chunk) >= 3] + \
            [chunk[3] for chunk in assignable_ally_unit_id_chunks if len(chunk) >= 4] + \
            [chunk[4] for chunk in assignable_ally_unit_id_chunks if len(chunk) >= 5] + \
            [chunk[5] for chunk in assignable_ally_unit_id_chunks if len(chunk) >= 6]

        platoon_unit_ids = \
            [chunk[6] for chunk in assignable_ally_unit_id_chunks if len(chunk) >= 7] + \
            [chunk[7] for chunk in assignable_ally_unit_id_chunks if len(chunk) >= 8] + \
            [chunk[8] for chunk in assignable_ally_unit_id_chunks if len(chunk) >= 9]
        scout_unit_ids = \
            [chunk[9] for chunk in assignable_ally_unit_id_chunks if len(chunk) >= 10]

        # Assign first units to move out at strategig angles to capture territory early game
        fixed_formation_unit_ids = [uid for uid in fixed_formation_ally_unit_ids if uid in alive_ally_unit_ids]
        moves.update(self.fixed_formation_moves(fixed_formation_unit_ids, [45, 20, 70]))

        # Assign a large portion of our units as sentinels
        moves.update(self.sentinel_moves([uid for uid in sentinel_unit_ids if uid in alive_ally_unit_ids]))

        # Assign a smaller chunk of our units to platoons
        moves.update(self.platoon_moves([uid for uid in platoon_unit_ids if uid in alive_ally_unit_ids]))

        # Assign the rest of our units to be scouts
        moves.update(self.scout_moves([uid for uid in scout_unit_ids if uid in alive_ally_unit_ids], map_states))
        #moves.update(self.scout_moves([uid for uid in alive_ally_unit_ids], map_states))

        return list(moves.values())

    #what is the enemy_count team_count for a given point
    def enemy_count_in_region(self):
        count = defaultdict(lambda: 0)
        enemy_in_region = {}
        for u, pt in self.enemy_units.items():

            if u in self.enemy_in_region:
                region = self.enemy_in_region[u]
                if region.detection_polygon.contains(pt):
                    count[region] += 1
                    enemy_in_region[u] = region
                    continue
            else:
                for region in self.regions:
                    if region.detection_polygon.contains(pt):
                        count[region] += 1
                        self.enemy_in_region[u] = region
                        break
        
        self.enemy_in_region = enemy_in_region
        return count

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

        return danger_score

    def wall_forces(self, current_point) -> List[Tuple[Tuple[float, float], float]]:
        current_x = current_point.x
        current_y = current_point.y
        dist_to_top = Point(current_x, 0).distance(current_point)
        dist_to_bottom = Point(current_x, 100).distance(current_point)
        dist_to_right = Point(0, current_y).distance(current_point)
        dist_to_left = Point(100, current_y).distance(current_point)
        return [((current_x, 0), dist_to_top), ((current_x, 100), dist_to_bottom), ((100, current_y), dist_to_right), ((0, current_y), dist_to_left)]

    def closest_friend_force(self, current_unit) -> List[Tuple[Tuple[float, float], float]]:
        current_pos = self.ally_units[current_unit]

        if(len(list(self.ally_units.keys())) < 2):
            return None

        closest_unit_dist = math.inf
        for friend_unit, friend_unit_pos in self.ally_units.items():
            if friend_unit == current_unit:
                continue
            dist = friend_unit_pos.distance(current_pos)
            if dist < closest_unit_dist:
                closest_unit_dist = dist
        return [((friend_unit_pos.x, friend_unit_pos.y), closest_unit_dist)]

    def least_popular_region_force(self, map_states):
        number_regions = len(self.entire_board_regions)
        unit_per_region = np.zeros(number_regions)
        unclaimed_regions = [region_id for region_id in self.scout if self.scout[region_id] is None]
        #print("LESAT POP")
        if not unclaimed_regions:
            '''
            for index in range(number_regions):
                current_poly = self.entire_board_regions[index]
                for player_num in range(4):
                    for unit in unit_pos[player_num]:
                        if current_poly.contains(unit):
                            unit_per_region[index] += 1

            index_min_region = int(np.argmin(unit_per_region))
            min_poly = self.entire_board_regions[index_min_region]
            center = min_poly.centroid
            # print(center)
            return [(index_min_region, (center.x, center.y))]
            '''
            return [(None, None)]

        for index in range(number_regions):
            if index in unclaimed_regions:
                current_poly_bounds = self.entire_board_region_tuples[index]
                for ux, uy in self.all_unit_pos_tuples:
                    minx, miny, maxx, maxy = current_poly_bounds
                    if (ux <= maxx and ux >= minx) and (uy <= maxy and uy >= miny):
                        unit_per_region[index] += 1
            else:
                unit_per_region[index] = math.inf

        #try to find non-occulded path where other
        # try to find non-occulded path
        non_occluded_path = False
        index_min_region = int(np.argmin(unit_per_region))
        min_poly_center = self.entire_board_region_centroids[index_min_region]
        for i in range(len(unit_per_region +1)):
            if self.path_home(min_poly_center, map_states):
                non_occluded_path = True
                break
        if non_occluded_path:
            return [(index_min_region, (min_poly_center.x, min_poly_center.y))]

        #otherwise, no occluded path, return sparsest region
        return [(index_min_region, (min_poly_center.x, min_poly_center.y))]

    def path_home(self, center, map_states):
        path = True
        center_coords = (center.x, center.y*-1)
        m = (self.home_coord_tuple[1] - center_coords[1]) / (self.home_coord_tuple[0] - center_coords[0])
        b = self.home_coord_tuple[1] - m*self.home_coord_tuple[0]

        if self.home_coord_tuple[0] > center_coords[0]:
            start_x = math.floor(center_coords[0])
            end_x = math.floor(self.home_coord_tuple[0])
        else:
            start_x = math.floor(self.home_coord_tuple[0])
            end_x = math.floor(center_coords[0])
        if start_x < 0:
            start_x = 0
        if end_x > 99:
            end_x = 99
        for i in range(start_x, end_x+1):
            current_y = m*i + b
            floored_y = -1*math.floor(current_y)
            if floored_y > 99:
                floored_y = 99
            elif floored_y < 0:
                floored_y = 0
            current_cell_state = map_states[i][floored_y]
            if current_cell_state != self.player_idx+1:
                path = False
                return

        return path

    def get_forces(self, unit_ids, map_states):
        forces = {id: [] for id in unit_ids}
        for unit, current_pos in [(uid, pos) for uid, pos in self.ally_units.items() if uid in unit_ids]:
            forces[unit].append([self.home_coord_tuple, self.home_coords.distance(current_pos)])
            forces[unit].append(self.wall_forces(current_pos))
            forces[unit].append(self.closest_friend_force(unit))
        return forces