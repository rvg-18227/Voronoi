import numpy as np
import logging
import warnings
from typing import Tuple, Optional
from collections import defaultdict

import shapely.errors
import shapely.geometry
import shapely.ops
import scipy


warnings.filterwarnings("ignore", category=shapely.errors.ShapelyDeprecationWarning)


def delaunay2edges(tri_simplices):
    """Convert the delaunay tris to unique edges
    Args:
        tri_simplices: Triangles. The .simplices param of the triangulation object from scipy.
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
                       poly2: shapely.geometry.polygon.Polygon) -> Optional[shapely.geometry.LineString]:
    """Return a line if polygons are neighbors. Polygons are neighbors iff they share an edge.
    Only 1 vertex does not count."""
    line = poly1.intersection(poly2)
    if isinstance(line, shapely.geometry.linestring.LineString):
        return line
    else:
        return None


class Player:
    def __init__(
        self, rng: np.random.Generator, logger: logging.Logger, total_days: int, spawn_days: int,
        player_idx: int, spawn_point: shapely.geometry.Point, min_dim: int, max_dim: int, precomp_dir: str
    ) -> None:
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

        self.home_offset = 0.5

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

        map_size = self.max_dim

        _MAP_W = map_size
        spawn_loc = {
            0: (self.home_offset, self.home_offset),
            1: (_MAP_W - self.home_offset, self.home_offset),
            2: (_MAP_W - self.home_offset, _MAP_W - self.home_offset),
            3: (self.home_offset, _MAP_W - self.home_offset)
        }

        # Construct 2 lists: triangulation/voronoi takes discrete, strategy takes continuous position
        # Note: Discretized points will have duplicates, which are removed (disputed points, both removed).
        discrete_pt2player, all_points = self.create_pts_player_dict(unit_pos)

        discrete_pts = list(discrete_pt2player.keys())
        discrete_players = list(discrete_pt2player.values())

        # Get polygon of Voronoi regions around each pt
        vor_regions = self.create_voronoi_regions(discrete_pts, map_size)

        # Find mapping from pts idx to polys (via nearest unit) and poly to player
        pt_to_poly, poly_idx_to_player = self.create_pt_to_poly_and_poly_idx_to_player(
            discrete_pts, vor_regions, discrete_players, spawn_loc)

        # Get the graph of connected pts via triangulation
        tri = scipy.spatial.Delaunay(np.array(discrete_pts))
        edges = delaunay2edges(tri.simplices)  # Shape: [N, 2]

        # Clean edges
        edges, edge_player_id = self.clean_edges(edges, discrete_players, discrete_pts, pt_to_poly, vor_regions,
                                                 poly_idx_to_player)

        # Create adjacency list for graph of armies
        adj_dict = self.create_adj_dict(edges, discrete_pts)

        moves = self.play_aggressive(vor_regions, all_points, pt_to_poly, adj_dict)
        # if self.current_day <= (50 - self.spawn_days) or total_scores[self.player_idx] < max(total_scores):
        #     # Create union of all friendly polygons and list of its neighbors
        #     superpolygon, s_neighbors = self.create_superpolygon(vor_regions, poly_idx_to_player, adj_dict)
        #     moves = self.play_aggressive(vor_regions, pt_player_dict, pt_to_poly, adj_dict, superpolygon,
        #                                  s_neighbors)
        # else:
        #     moves = self.play_cautious(unit_id, unit_pos, vor_regions, pt_player_dict, pt_to_poly, adj_dict,
        #                                superpolygon, s_neighbors)

        self.current_day += 1
        return moves

    def create_pts_player_dict(self, units) -> tuple[dict[tuple, int], dict[int, list[tuple]]]:
        """Return all quantized non-disputed units and the player they correspond to.

        Returns:
            dict: Pt to player map {pt: player}
            dict: Player to all points map
        """
        # TODO: Use occ map to remove disputed points, and get player for each remaining point
        pts_hash = {}
        disputed_pts = []
        all_points = defaultdict(lambda: [])
        for pl in range(4):
            for spt in units[pl]:
                # Add point to all units list regardless
                x, y = spt.coords[0]
                all_points[pl].append((x, y))

                # Quantize unit pos to cell. We assume cell origin at center.
                pos_int = (int(x) + self.home_offset, int(y) + self.home_offset)

                if pos_int in pts_hash:
                    player_existing = pts_hash[pos_int]
                    if player_existing != pl:
                        # Disputed cell - remove later
                        disputed_pts.append(pos_int)
                else:
                    pts_hash[pos_int] = pl

        for pos_int in disputed_pts:
            pts_hash.pop(pos_int)

        return pts_hash, all_points

    def create_adj_dict(self, edges, pts):
        adj_dict = defaultdict(lambda: [])
        for val in edges:
            p1 = pts[val[0]]
            p2 = pts[val[1]]
            # Adjacency = bi-directional graph
            adj_dict[p1].append(p2)
            adj_dict[p2].append(p1)

        return adj_dict

    def create_voronoi_regions(self, pts, map_size):
        # Get polygon of Voronoi regions around each pt
        _points = shapely.geometry.MultiPoint(pts)
        envelope = shapely.geometry.box(0, 0, map_size, map_size)
        vor_regions_ = shapely.ops.voronoi_diagram(_points, envelope=envelope)
        vor_regions_ = list(vor_regions_)  # Convert to a list of Polygon

        # The polys aren't being bounded correctly. Fix manually.
        # TODO: Shortlist poly candidates by seeing which ones have coords outside bounds.
        vor_regions = []
        for region in vor_regions_:
            if not isinstance(region, shapely.geometry.Polygon):
                print(f"WARNING: Region returned from voronoi not a polygon: {type(region)}")

            region_bounded = region.intersection(envelope)
            if region_bounded.area > 0:
                vor_regions.append(region_bounded)

        return vor_regions

    def create_pt_to_poly_and_poly_idx_to_player(self, pts, vor_regions, player_ids, spawn_loc):
        # TODO: Optimize
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

        return pt_to_poly, poly_idx_to_player

    def clean_edges(self, edges, player_ids_with_home, pts_with_home, pt_to_poly, vor_regions, poly_idx_to_player):
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
                if are_neighbors is None:  # and play1_ == player1 and play2_ == player1:
                    # Can traverse edge only if voronoi polys are neighbors
                    edge_player_id.append(-2)
                    continue

                if play1_ == player1 and play2_ == player1:
                    valid_ = True

            if valid_:
                edge_player_id.append(player1)
            else:
                edge_player_id.append(-1)

        edge_player_id = np.array(edge_player_id)
        edges = edges[edge_player_id > -2]  # remove edges that are not neighbors
        edge_player_id = edge_player_id[edge_player_id > -2]

        return edges, edge_player_id

    def create_superpolygon(self, vor_regions, poly_idx_to_player, adj_dict):
        friendly_polygons = []
        for idx, reg in enumerate(vor_regions):
            if poly_idx_to_player[idx] == self.player_idx:
                friendly_polygons.append(reg)

        superpolygon = shapely.ops.unary_union(friendly_polygons)

        s_neighbors = set()
        for unit in adj_dict:
            if superpolygon.contains(shapely.geometry.Point(unit[0], unit[1])):
                for neigh in adj_dict[unit]:
                    if not superpolygon.contains(shapely.geometry.Point(neigh[0], neigh[1])):
                        s_neighbors.add(neigh)
        s_neighbors = list(s_neighbors)

        return superpolygon, s_neighbors

    def find_target_from_superpolygon(self, unit, superpolygon, s_neighbors, friendly_units, pt_to_poly, vor_regions):
        neighboring_enemies = [n for n in s_neighbors if n not in friendly_units]
        neighboring_enemy_polygons = [pt_to_poly[ne] for ne in neighboring_enemies]

        candidates = set()
        for poly_idx in neighboring_enemy_polygons:
            polygon = vor_regions[poly_idx]
            intersection = superpolygon.intersection(polygon)
            if isinstance(intersection, shapely.geometry.LineString):
                points = intersection.coords
                for point in list(points):
                    candidates.add(point)

        if len(candidates) == 0:
            target = unit  # if no enemies are found, stay in the same place
        else:
            target = max(list(candidates), key=lambda pt: (pt[0] - unit[0]) ** 2 + (pt[1] - unit[1]) ** 2)

        return target

    def move_toward_position(self, current, target):
        # TODO: what if we always pass a distance of 1?
        distance_to_target = np.sqrt((target[0] - current[0]) ** 2 + (target[1] - current[1]) ** 2)

        if distance_to_target == 0:
            angle_toward_target = 0.0
        else:
            angle_toward_target = np.arctan2(target[1] - current[1], target[0] - current[0])

        return max(1.0, distance_to_target), angle_toward_target

    # def play_cautious(self, unit_id, unit_pos, vor_regions, pt_player_dict, pt_to_poly, adj_dict, superpolygon,
    #                   s_neighbors):
    #     moves = self.play_aggressive(vor_regions, pt_player_dict, pt_to_poly, adj_dict, superpolygon, s_neighbors)
    #     # TODO: Better logic - we don't know that the last 4 units are closes to home (or last to spawn)
    #     fort_positions = [(0.6, 1.6), (1.6, 0.6), (1.6, 1.6)]
    #
    #     for idx in range(len(fort_positions)):
    #         current_point = unit_pos[self.player_idx][-idx]
    #         current = (current_point.x, current_point.y)
    #         target = fort_positions[idx]
    #         move = self.move_toward_position(current, target)
    #         moves[-idx] = move
    #
    #     return moves

    def play_aggressive(self, vor_regions, all_points, pt_to_poly, adj_dict):
        moves = []

        # Generate a move for every unit
        friendly_units = all_points[self.player_idx]

        for unit_ in friendly_units:
            # All polys, etc are indexed from cell origin
            x, y = unit_
            unit = (int(x) + self.home_offset, int(y) + self.home_offset)
            current_polygon = vor_regions[pt_to_poly[unit]]

            neighbors = adj_dict[unit]
            neighboring_enemies = [n for n in neighbors if n not in friendly_units]
            neighboring_enemy_polygons = [pt_to_poly[ne] for ne in neighboring_enemies]

            if len(neighboring_enemies) == 0:
                # Moving to centroid will spread units out
                target = current_polygon.centroid.coords[0]
            else:
                # Strategy - Move to furthest vertex of neighboring edges
                candidates = set()
                for poly_idx in neighboring_enemy_polygons:
                    polygon = vor_regions[poly_idx]
                    intersection = current_polygon.intersection(polygon)
                    if isinstance(intersection, shapely.geometry.LineString):
                        points = intersection.coords
                        for point in list(points):
                            candidates.add(point)

                # further enemy polygon vertex on shared edges
                if len(candidates) > 0:
                    target = max(list(candidates), key=lambda pt: (pt[0] - unit[0]) ** 2 + (pt[1] - unit[1]) ** 2)
                else:
                    # no neighboring enemies
                    target = current_polygon.centroid.coords[0]

                # # Strategy - Move to middle of edge - closest edge
                # candidates = set()
                # for poly_idx in neighboring_enemy_polygons:
                #     polygon = vor_regions[poly_idx]
                #     intersection = current_polygon.intersection(polygon)
                #     if isinstance(intersection, shapely.geometry.LineString):
                #         points = intersection.coords[:]  # tuple
                #         points_np = np.array(points)
                #         edge_center = tuple(points_np.mean(axis=0).tolist())
                #         candidates.add(edge_center)
                #
                # min_dist = 1e3
                # pt = None
                # for edge_c in candidates:
                #     dist = np.linalg.norm(np.array(edge_c) - np.array(unit))
                #     if dist < min_dist:
                #         min_dist = dist
                #         pt = edge_c
                # if pt is None:
                #     # The neighboring enemies don't share an edge with this unit
                #     # todo: clean the edges
                #     pt = current_polygon.centroid.coords[0]
                # target = pt

                # # Strategy - Move to mean of neighboring enemies
                # neig_ene = np.array(neighboring_enemies)  # (N, 2)
                # neig_ene_mean = neig_ene.mean(axis=0)
                # target = tuple(neig_ene_mean.tolist())

            # Note: Earlier, movement angle was calculated from cell center,
            #   not the actual position of the unit
            moves.append(self.move_toward_position(unit, target))

        return moves
