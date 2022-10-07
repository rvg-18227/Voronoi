import logging
import math
import os
import pickle
from typing import Tuple, List

import numpy as np
from shapely.geometry import Point
import sympy


WALL_DENSITY = 0.1
WALL_RATIO = 0


def midsort(arr: list[float]) -> list[float]:
    first_elem_added = False

    n = len(arr)
    if n <= 2:
        return arr

    prev_midpoints = [0, n - 1]
    midsorted_arr = []

    while len(prev_midpoints) < n:
        curr_midpoints = []

        for i, (left_pt, right_pt) in enumerate(zip(prev_midpoints, prev_midpoints[1:])):
            mid_pt = (left_pt + right_pt) // 2
            if mid_pt != left_pt or mid_pt == 0:
                curr_midpoints.extend([left_pt, mid_pt])
                midsorted_arr.append(arr[mid_pt])

                if mid_pt == 0:
                    first_elem_added = True

            else:
                curr_midpoints.append(left_pt)

            if i == len(prev_midpoints) - 2:
                curr_midpoints.append(right_pt)

        prev_midpoints = curr_midpoints

    # add the LAST element in the original @arr
    if not first_elem_added:
        midsorted_arr.append(arr[0])
        
    midsorted_arr.append(arr[-1])

    return midsorted_arr

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
        
        self.us = player_idx
        self.homebase = np.array(spawn_point)

        # print(player_idx, spawn_point)

        self.target_loc = []

        self.initial_radius = 25

        base_angle = (-1) * ((math.pi / 2) + (player_idx * math.pi/2))

        #  (base angle + pi/2) 
        #    ^
        #    | 
        #    |     
        #    -------> (base angle): -pi/2 - pi/2 * (player_index)
        # p3
        outer_wall_angles = np.linspace(start=base_angle, stop=(base_angle +
            math.pi/2), num=(total_days // spawn_days))
        self.midsorted_outer_wall_angles = midsort(outer_wall_angles)

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

        while len(unit_id[self.us]) > len(self.target_loc):
            # add new target_locations
            self.target_loc.append(
                self.order2coord([self.initial_radius, self.midsorted_outer_wall_angles[len(unit_id[self.us]) - 1]]))
        
        return get_moves(shapely_pts_to_tuples(unit_pos[self.us]), self.target_loc)

    def order2coord(self, order) -> tuple[float, float]:
        dist, angle = order
        x = self.homebase[0] - dist * math.sin(angle)
        y = self.homebase[1] + dist * math.cos(angle)
        return (x, y)


def get_moves(unit_pos, target_loc) -> list[tuple[float, float]]:
    assert len(unit_pos) == len(target_loc), "get_moves: unit_pos and target_loc array length not the same"
    np_unit_pos = np.array(unit_pos, dtype=float)
    np_target_loc = np.array(target_loc, dtype=float)

    cord_diff = np_target_loc - np_unit_pos
    cord_diff_x = cord_diff[:, 0]
    cord_diff_y = cord_diff[:, 1]

    move_dist = np.linalg.norm(cord_diff, axis=1)
    move_dist[move_dist > 1] = 1.0
    move_angle = np.arctan2(cord_diff_y, cord_diff_x)
    
    move_arr = list(zip(move_dist, move_angle))
    return move_arr

def shapely_pts_to_tuples(points: List[Point]) -> List[Tuple[float, float]]:
    return list(map(shapely_pt_to_tuple, points))

def shapely_pt_to_tuple(point: Point) -> Tuple[float, float]:
    return ( float(point.x), float(point.y) )


def test_midsort():
    cases = [
        {
            'name': 'array_of_odd_number_of_elements',
            'array': [1, 2, 3, 4, 5],
            'expect': [3, 2, 4, 1, 5]
        },
        {
            'name': 'corner_case_empty_array',
            'array': [],
            'expect': []
        },
        {
            'name': 'array_of_single_element',
            'array': [10],
            'expect': [10]
        },
        {
            'name': 'array_of_2_elements',
            'array': [10, 30],
            'expect': [10, 30]
        },
        {
            'name': 'array_of_even_number_of_elements',
            'array': [10, 78, 290, 208, 284, 285, 203, 173],
            'expect': [208, 78, 285, 10, 290, 284, 203, 173]
        }
    ]

    error_count = 0

    for tc in cases:
        got = midsort(tc['array'])

        if got != tc['expect']:
            print(f'case {tc["name"]} failed:')
            print(f'expect: {tc["expect"]}')
            print(f'got: {got}\n')
            error_count += 1
        
    
    if error_count == 0:
       print("PASSED - test_midsort")
    else:
       print(f"FAILED with {error_count} errors - test_midsort")
        

def test_get_moves():
    cases = [
        {
            "unit_pos": [[0, 0], [0, 0], [0, 0]],
            "target_loc": [[1, 1], [1, 4], [2, 5]]
        }
    ]
    
    result = get_moves(cases[0]['unit_pos'], cases[0]['target_loc'])
    print("radians: " + str(result))

    for i in range(len(result)):
        result[i] = [result[i][0], result[i][1] * 180 / math.pi]
    
    print("degrees: " + str(result))
