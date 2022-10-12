import os
import pickle
import numpy as np
import logging
from typing import Tuple
import math
import sympy
from sympy import Point,Polygon
import shapely.geometry
"""Player module for Group 8 probabilistic model - Voronoi."""

class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger, total_days: int, spawn_days: int,
                 player_idx: int, spawn_point: shapely.geometry.Point, min_dim: int, max_dim: int, precomp_dir: str) \
            -> None:
        """Initialise the player with given skill.

            Args:
                rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
                logger (logging.Logger): logger use this like logger.info("message")
                total_days (int): total number of days, the game is played
                spawn_days (int): number of days after which new units spawn
                player_idx (int): index used to identify the player among the four possible players
                spawn_point (Tuple): Homebase of the player. Shape: (2,)
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
        self.parts_angle = []
        self.cur_unit = 1
        self.n = 6 ##how many direction do we want the points to look at 

    def play(self, unit_id, unit_pos, map_states, current_scores, total_scores) -> [Tuple[float, float]]:
        """Function which based on current game state returns the distance and angle of each unit active on the board

                Args:
                    unit_id (list(list(str))): contains the ids of each player's units (unit_id[player_idx][x])
                    unit_pos (list(list(shapely.geometry.Point))): contains the position of each unit currently present
                                                    on the map (unit_pos[player_idx][x])
                    map_states (list(list(int)): contains the state of each cell, using the x, y coordinate system
                                                    (map_states[x][y])
                    current_scores (list(int)): contains the number of cells currently occupied by each player
                                                    (current_scores[player_idx])
                    total_scores (list(int)): contains the cumulative scores up until the current day
                                                    (total_scores[player_idx]

                Returns:
                    List[Tuple[float, float]]: Return a list of tuples consisting of distance and angle in radians
                        to move each unit of the player.
                """

        moves = []
        if self.player_idx >1:
            self.cur_unit = max(int(unit_id[self.player_idx][-1]),self.cur_unit) ## grab the last id value which is also the number of units present
        points = unit_pos[self.player_idx]
        self.points =  list(map(np.array,unit_pos[self.player_idx]))
        self.enemy_position = []
        self.map_states = map_states
        for i in range(4):
            if i == (self.player_idx-1):
                continue
            self.enemy_position+=  list(map(np.array,unit_pos[i])) ## add all the other player's position into a list
        
        for i in range(0,720,720//self.n):
            self.parts_angle.append(math.radians(i)) ##angle are always in randians!!!!!!
        for point in points:
            direction = self.get_direction(point)
            distance = 1
            moves.append((direction,distance))
        return moves
    def get_direction(self,point:list[float]):
        
        direction_score = []
        for i in range(len(self.parts_angle)-1):
            distance = 5
            angle1 = self.parts_angle[i]
            angle2 = self.parts_angle[i]
            point = np.array(point)
            ## find all the points that are encircled by the angle
            p1 = self.get_point(point,angle1,distance=5)
            p2 = self.get_point(point,angle2,distance=5)
            
            
            #heuristic we are considering
            #edge within 5 blocks
            edge_score = self.find_edge_score(point,angle1,angle2,distance) 
            
            # puts the point back onto the map ( basically a triangle)
            #enemy within 5 blocks
            enemy_score = self.find_enemy_score(point,angle1,angle2,distance)
            #ally within 5 blocks
            ally_score = self.find_ally_score(point,angle1,angle2,distance)
            #Free open space within 10 blocks( using other player's coordinate)
            open_space_score = self.find_open_space_score(point,angle1,angle2,distance)

            total_score = (edge_score*-1+ open_space_score  + (enemy_score*-1  + ally_score))/4
            direction_score.append(total_score)

        
        norm_direction = [float(i)/sum(direction_score) for i in direction_score]
        index = self.rng.choice(range(len(norm_direction)), p = norm_direction)
        within = self.rng.random()* 720/self.n #choose within the area
        direction  = self.parts_angle[index] + within
        return direction
    def checkboundary(self,point:list[float])->list[float]:
        x,y = point
        new_x = x
        new_y = y
        if x>= 100:
            new_x = 100
        if y >= 0:
            new_y = 100
        if x<= 0:
            new_x = 0
        if y <= 0:
            new_y = 0
        return [new_x,new_y]

    def get_point(self,point:list[float],angle:float,distance:int)-> list[float]:
        ##given a point and an angle, provide where the point will land
        x,y = point
        x_val = x + math.cos(angle)*distance
        y_val = y + math.sin(angle)*distance
        return [x_val,y_val]
    def find_edge_score(self,point:list[float],angle1,angle2,distance) -> float:
        ##the area exceeding the boundary/ the area we are looking 
        p1 = self.get_point(point,angle1,distance)
        p2 = self.get_point(point,angle2,distance)

        w1, w2, w3, w4 = map(Point,[[0,0],[0,100],[100,0],[100,100]])
        t1,t2,t3 =  map(Point,[point,p1,p2])
        world = sympy.Polygon(w1, w2, w3, w4)
        triangle = sympy.Polygon(t1,t2,t3)
        intersecected_points = triangle.intersection(world)
        if len(intersecected_points)<3:
            return 0
        overlap = shapely.geometry.Polygon(np.array(intersecected_points)).area
        triangle_area = shapely.geometry.Polygon([point,p1,p2]).area
        score = overlap/triangle_area *100 # out of 100 % normalize
        return score
    

    def find_enemy_score(self,point:list[float],angle1,angle2,distance) -> int:
        # the number of enemy enclosed / the current number of enemies
        p1 = self.get_point(point,angle1,distance)
        p2 = self.get_point(point,angle2,distance)
        p1 = self.checkboundary(p1)
        p2 = self.checkboundary(p2)
        high_x = max(p1[0],p2[0],point[0])
        high_y = max(p1[1],p2[1],point[1])
        low_x = min(p1[0],p2[0],point[0])
        low_y = min(p1[1],p2[1],point[1])
        num_enemy_enclosed = 0
        for enemy in self.enemy_position:
            enemy_x = enemy[0]
            enemy_y = enemy[1]
            if low_x<= enemy_x<=high_x and low_y<= enemy_y<=high_y:
                num_enemy_enclosed+=1
        return num_enemy_enclosed/(self.cur_unit*3) #normalize
    
    def find_ally_score(self,point:list[float],angle1,angle2,distance) -> float:
        p1 = self.get_point(point,angle1,distance)
        p2 = self.get_point(point,angle2,distance)
        p1 = self.checkboundary(p1)
        p2 = self.checkboundary(p2)
        high_x = max(p1[0],p2[0],point[0])
        high_y = max(p1[1],p2[1],point[1])
        low_x = min(p1[0],p2[0],point[0])
        low_y = min(p1[1],p2[1],point[1])
        num_ally_enclosed = 0
        for ally in self.points:
            ally_x = ally[0]
            ally_y = ally[1]
            if low_x<= ally_x<=high_x and low_y<= ally_y<=high_y:
                num_ally_enclosed+=1
        return num_ally_enclosed/self.cur_unit # normalize
    def find_open_space_score(self,point:list[float],angle1,angle2,distance)-> float:
        p1 = self.get_point(point,angle1,distance)
        p2 = self.get_point(point,angle2,distance)
        p1 = self.checkboundary(p1)
        p2 = self.checkboundary(p2)
        np_map = np.array(self.map_states)
        
        high_x = max(p1[0],p2[0],point[0],0)
        high_y = max(p1[1],p2[1],point[1],0)
        low_x = min(p1[0],p2[0],point[0],100)
        low_y = min(p1[1],p2[1],point[1],100)

        y_range = list(range(math.floor(low_y),math.ceil(high_y)))
        x_range = list(range(math.floor(low_x),math.ceil(high_x)))
        np_map = np_map[:,y_range]
        np_map = np_map[x_range,:]
        not_yet_occupied_cell = 0
        for x in range(math.floor(low_x),math.ceil(high_x)):
            for y in range(math.floor(low_y),math.ceil(high_y)):
                if self.map_states[x][y] != self.player_idx and self.map_states[x][y] != -1:
                    not_yet_occupied_cell +=1
        possible_area = math.ceil(high_x)-math.floor(low_x) * (math.ceil(high_x)-math.floor(low_x))
        return not_yet_occupied_cell/(abs(possible_area))



    def safety_heuristic(self,point:list[float],rad)->list[float]:
        ##point and how far we want to look 
        num_enemy_near = 0
        num_ally_near = 0
        for enemy in self.enemy_position:
            enemy_x = enemy[0]
            enemy_y = enemy[1]
            if  self.isInside(point[0], point[1], rad, enemy_x, enemy_y): num_enemy_near+=1
        for ally in self.points:
            ally_x = ally[0]
            ally_y = ally[1]
            if  self.isInside(point[0], point[1], rad, ally_x, ally_y): num_ally_near+=1
        return num_enemy_near,num_ally_near
    



    def isInside(self,circle_x, circle_y, rad, x, y):
        
        # Compare radius of circle
        # with distance of its center
        # from given point
        if ((x - circle_x) * (x - circle_x) +
            (y - circle_y) * (y - circle_y) <= rad * rad):
            return True
        else:
            return False
 
 