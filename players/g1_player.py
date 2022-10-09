import os
import pickle
from pty import spawn
import py_compile
from turtle import distance
import numpy as np
import sympy
import logging
from typing import Tuple
from collections import defaultdict

import matplotlib as mpl
import shapely.errors
import shapely.geometry
import shapely.ops
import scipy
from matplotlib import pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=shapely.errors.ShapelyDeprecationWarning)

def delaunay2edges(tri_simplices):
    """Convert the delaunay tris to unique edges
    Args:
        simplices: Triangles. The .simplices param of the triangulation object from scipy.
    Ref:
        https://stackoverflow.com/questions/69512972/how-to-generate-edge-index-after-delaunay-triangulation
    """

    def less_first(a, b):
        return (a, b) if a < b else (b, a)

    edges_dict = defaultdict(list)  # Gives all the edges and the associated triangles
    for idx, triangle in enumerate(tri_simplices):
        for e1, e2 in [[0, 1], [1, 2], [2, 0]]:  # for all edges of triangle
            edge = less_first(triangle[e1], triangle[e2])  # always lesser index first
            edges_dict[edge].append(idx)  # prevent duplicates. Track associated simplexes for each edge.

    array_of_edges = np.array(list(edges_dict.keys()), dtype=int)
    return array_of_edges

def poly_are_neighbors(poly1: shapely.geometry.polygon.Polygon,
                       poly2: shapely.geometry.polygon.Polygon) -> bool:
    # Polygons are neighbors iff they share an edge. Only 1 vertex does not count.
    # Also, both polygons might be the same
    if isinstance(poly1.intersection(poly2), shapely.geometry.linestring.LineString):
        return True
    elif poly1 == poly2:
        return True
    else:
        return False

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
        self.precomp_dir = precomp_dir

        self.player_idx = player_idx
        self.spawn_point = spawn_point
        self.min_dim = min_dim
        self.max_dim = max_dim

        self.total_days = total_days
        self.spawn_days = spawn_days
        self.current_day = 1

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

        map_size = 100
        home_offset = 0.5
        _MAP_W = map_size
        spawn_loc = {0: (home_offset, home_offset),
                    1: (_MAP_W - home_offset, home_offset),
                    2: (_MAP_W - home_offset, _MAP_W - home_offset),
                    3: (home_offset, _MAP_W - home_offset)}

        # Construct 2 lists for passing to shapely/scipy for triangulation
        # Handle edge case where cell is occupied by 2 players: Quantize pts to grid cells.
        #   When a cell is disputed, it no longer contributes to the voronoi diagram. All the units within that cell must
        #   be removed.
        units = []
        for player in range(len(unit_pos)):
            for pos in unit_pos[player]:
                units.append((player, (pos.x, pos.y)))

        pts_hash = {}
        for pl,pos in units:
            # Quantize unit pos to cell. We assume cell origin at center.
            pos_int = (int(pos[0]) + home_offset, int(pos[1]) + home_offset)

            if pos_int in pts_hash:
                player_existing = pts_hash[pos_int]
                if player_existing == pl:
                    pts_hash[pos_int] = pl
                else:
                    pass  # Disputed cell
            else:
                pts_hash[pos_int] = pl
        pts = list(pts_hash.keys())
        player_ids = list(pts_hash.values())
        pts_coords_hash = {coords: index for index, coords in enumerate(pts)}

        # Get polygon of Voronoi regions around each pt
        _points = shapely.geometry.MultiPoint(pts)
        envelope = shapely.geometry.box(0, 0, map_size, map_size)
        vor_regions_ = shapely.ops.voronoi_diagram(_points, envelope=envelope)
        vor_regions_ = list(vor_regions_)  # Convert to a list of Polygon

        # The polys aren't being bounded correctly. Fix manually.
        vor_regions = []
        for region in vor_regions_:
            if not isinstance(region, shapely.geometry.Polygon):
                print(f"WARNING: Region returned from voronoi not a polygon: {type(region)}")

            region_bounded = region.intersection(envelope)
            if region_bounded.area > 0:
                vor_regions.append(region_bounded)

        # # Add the home base to list of points
        # pts_with_home = pts.copy()
        # player_ids_with_home = player_ids.copy()
        # for player in range(4):
        #     # Add home bases as pts
        #     pts_with_home.append(spawn_loc[player])
        #     player_ids_with_home.append(player)

        # Find mapping from pts idx to polys (via nearest unit) and poly to player
        pt_to_poly = {}  # includes home base
        # Polygon isn't hashable, so we use polygon idx.
        poly_idx_to_player = {}
        kdtree = scipy.spatial.KDTree(pts)
        for region_idx, region in enumerate(vor_regions):
            # Voronoi regions are all convex. Nearest pt to centroid must be point belonging to region
            # centroid = region.centroid.coords[:][0]
            # _, ii = kdtree.query(centroid, k=1)  # index of nearest pt
            repr_pt = region.representative_point()
            _, ii = kdtree.query(repr_pt, k=1)  # index of nearest pt

            pt_to_poly[pts[ii]] = region_idx
            poly_idx_to_player[region_idx] = player_ids[ii]

        # Find mapping of each home base to poly
        for idx in range(4):
            home_coord = spawn_loc[idx]
            _, ii = kdtree.query(home_coord, k=1)  # index of nearest pt
            pt_to_poly[home_coord] = pt_to_poly[pts[ii]]  # home base same as nearest unit

        # Get the graph of connected pts via triangulation (include home base when triangulating)
        pts_with_home = pts.copy()
        player_ids_with_home = player_ids.copy()
        # for key, val in spawn_loc.items():
        #     # Add home bases as pts
        #     pts_with_home.append(val)
        #     player_ids_with_home.append(key)

        tri = scipy.spatial.Delaunay(np.array(pts_with_home))
        edges = delaunay2edges(tri.simplices)  # Shape: [N, 2]

        # Clean edges
        # TODO: Handle case when enemy unit within home cell. It will cut off all player's units.
        #  Soln: When making graph, remove edge.
        #  Problem: How will we do a path search to home base?
        
        edge_player_id = []  # Player each edge belongs to
        for idx, (p1, p2) in enumerate(edges):
            player1 = player_ids_with_home[p1]
            player2 = player_ids_with_home[p2]

            valid_ = False
            if player1 == player2:
                poly1_idx = pt_to_poly[tuple(pts_with_home[p1])]
                poly2_idx = pt_to_poly[tuple(pts_with_home[p2])]
                poly1 = vor_regions[poly1_idx]
                poly2 = vor_regions[poly2_idx]

                # The polygons must both belong to the same player
                # This handles edge cases where home base is conquered by another player
                play1_ = poly_idx_to_player[poly1_idx]
                play2_ = poly_idx_to_player[poly2_idx]

                are_neighbors = poly_are_neighbors(poly1, poly2)
                if not are_neighbors: # and play1_ == player1 and play2_ == player1:
                    # Can traverse edge only if voronoi polys are neighbors
                    edge_player_id.append(-2)
                    continue
                    # valid_ = True
            
                if play1_ == player1 and play2_ == player1:
                    valid_ = True

            if valid_:
                edge_player_id.append(player1)
            else:
                edge_player_id.append(-1)

        edge_player_id = np.array(edge_player_id)
        edges = edges[edge_player_id > -2] # remove edges that are not neighbors
        edge_player_id = edge_player_id[edge_player_id > -2]

        adj_dict = {}
        for val in edges:
            p1 = pts[val[0]]
            v1 = val[1]
            p2 = pts[v1]

            if p1 not in adj_dict: adj_dict[p1] = []
            if p2 not in adj_dict: adj_dict[p2] = []

            adj_dict[p1].append(p2)
            adj_dict[p2].append(p1)

        #######################################################################


        # TODO: BUILD A LIST OF NEIGHBORING POLYGONS FOR EVERY POLYGON

        # Build a graph data structure for each player
        # graphs = {0: defaultdict(list), 1: defaultdict(list), 2: defaultdict(list), 3: defaultdict(list)}
        # for player, (p1, p2) in zip(edge_player_id, edges):
        #     if player > -1:
        #         graph_p = graphs[player]
        #         graph_p[p1].append(p2)
        #         graph_p[p2].append(p1)

        # TODO: From each home base, traverse the full graph

        #######################################################################

        if self.current_day <= 50 or total_scores[self.player_idx] < max(total_scores):
            moves = self.play_aggressive(home_offset, vor_regions, units, pt_to_poly, adj_dict)
        else:
            moves = self.play_cautious(unit_id, unit_pos, home_offset, vor_regions, units, pt_to_poly, adj_dict)
        
        # moves = self.play_aggressive(home_offset, vor_regions, units, pt_to_poly, adj_dict)
        self.current_day += 1
        return moves

    def move_toward_position(self, current, target):
            distance_to_target = np.sqrt((target[0] - current[0])**2 + (target[1] - current[1])**2)

            if distance_to_target == 0:
                angle_toward_target = 0.0
            else:
                angle_toward_target = np.arctan2(target[1] - current[1], target[0] - current[0])
            
            return max(1.0, distance_to_target), angle_toward_target

    def play_cautious(self, unit_id, unit_pos, home_offset, vor_regions, units, pt_to_poly, adj_dict):
        moves = self.play_aggressive(home_offset, vor_regions, units, pt_to_poly, adj_dict)
        fort_unit_ids = unit_id[self.player_idx][-4:-1]
        fort_positions = [(0.5, 1.5), (1.5, 0.5), (1.5, 1.5)]

        for i in range(len(fort_positions)):
            current_point = unit_pos[self.player_idx][int(fort_unit_ids[i])]
            current = (current_point.x, current_point.y)
            target = fort_positions[i]
            move = self.move_toward_position(current, target)
            moves[int(fort_unit_ids[i])] = move

        return moves

    def play_aggressive(self, home_offset, vor_regions, units, pt_to_poly, adj_dict):
        moves = []

        # For each friendly unit, find the direction toward the farthest nearest enemy-bordering vertex
        friendly_units = [u for p,u in units if p == self.player_idx]
        friendly_units = [(int(x) + home_offset, int(y) + home_offset) for x,y in friendly_units]

        for unit in friendly_units:
            current_polygon = vor_regions[pt_to_poly[unit]]

            neighbors = adj_dict[unit]
            neighboring_enemies = [n for n in neighbors if n not in friendly_units]
            neighboring_enemy_polygons = [pt_to_poly[ne] for ne in neighboring_enemies]

            candidates = set() 
            for poly_idx in neighboring_enemy_polygons:
                polygon = vor_regions[poly_idx]
                intersection = current_polygon.intersection(polygon)
                if isinstance(intersection, shapely.geometry.LineString):
                    points = intersection.coords
                    for point in list(points):
                        candidates.add(point)

            # if current point is surrounded by friendly polygons, move toward center
            if len(candidates) == 0:
                target = (50.0, 50.0)
                # TODO: FIX THIS
            else:
                target = max(list(candidates), key=lambda pt: (pt[0] - unit[0])**2 + (pt[1] - unit[1])**2)

            moves.append(self.move_toward_position(unit, target))

        return moves
