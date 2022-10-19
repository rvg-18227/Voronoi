import numpy as np
import logging
import warnings
from typing import Tuple, Optional
from collections import defaultdict

import shapely.errors
import shapely.geometry
import shapely.ops
import scipy
from sklearn.cluster import DBSCAN

# from tests import plot_funcs
# from tests.plot_funcs import plot_units_and_edges, plot_poly_list, plot_incursions, plot_line_list, plot_debug_incur

warnings.filterwarnings("ignore", category=shapely.errors.ShapelyDeprecationWarning)


class CreateGraph:
    def __init__(self, home_offset):
        self.home_offset = home_offset

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
            if pos_int in pts_hash:
                pts_hash.pop(pos_int)

        return pts_hash, all_points

    @staticmethod
    def create_pt_to_poly_and_idx(kdtree, discrete_pts, vor_regions):
        # kdtree contains all the discrete points
        # Find mapping from pts idx to polys (via nearest unit) and poly to player
        pt_to_poly_idx = {}  # Polygon isn't hashable, so we use polygon idx.
        pt_to_poly = {}
        poly_idx_to_pt = {}

        for region_idx, region in enumerate(vor_regions):
            # Voronoi regions are all convex. Nearest pt to centroid must be point belonging to region
            centroid = region.centroid.coords[0]
            _, ii = kdtree.query(centroid, k=1)  # index of nearest pt
            # repr_pt = region.representative_point()
            # _, ii = self.kdtree.query(repr_pt, k=1)  # index of nearest pt

            pt_to_poly_idx[discrete_pts[ii]] = region_idx
            pt_to_poly[discrete_pts[ii]] = region
            poly_idx_to_pt[region_idx] = discrete_pts[ii]

        return pt_to_poly_idx, pt_to_poly, poly_idx_to_pt

    @staticmethod
    def delaunay2edges(tri_simplices) -> np.ndarray:
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

    @staticmethod
    def poly_are_neighbors(poly1: shapely.geometry.polygon.Polygon,
                           poly2: shapely.geometry.polygon.Polygon) -> Optional[shapely.geometry.LineString]:
        """Return a line if polygons are neighbors. Polygons are neighbors iff they share an edge.
        Only 1 vertex does not count."""
        line = poly1.intersection(poly2)
        if isinstance(line, shapely.geometry.linestring.LineString):
            return line
        else:
            return None

    @staticmethod
    def create_adj_dict(edges, pts):
        adj_dict = defaultdict(lambda: [])
        for val in edges:
            p1 = pts[val[0]]
            p2 = pts[val[1]]
            # Adjacency = bi-directional graph
            adj_dict[p1].append(p2)
            adj_dict[p2].append(p1)

        return adj_dict

    @staticmethod
    def create_voronoi_regions(pts: list[tuple], map_size: int):
        """Get polygon of Voronoi regions around each pt. Bound polygons to map.

        Args:
            pts: all units
            map_size: Size of map

        Returns:
            list: List of polygons (not in same order as input points)
            list: List of ints. 1 if corresponding poly was corrected (was out of bounds), 0 otherwise
        """
        #
        _points = shapely.geometry.MultiPoint(pts)
        envelope = shapely.geometry.box(0, 0, map_size, map_size)
        vor_regions_ = shapely.ops.voronoi_diagram(_points, envelope=envelope)
        vor_regions_ = list(vor_regions_)  # Convert to a list of Polygon

        # The polys aren't bounded. Fix manually.
        vor_regions = []
        region_was_unbounded = []  # Whether a region was out of bounds
        for region in vor_regions_:
            # if not isinstance(region, shapely.geometry.Polygon):
            #     print(f"WARNING: Region returned from voronoi not a polygon: {type(region)}")

            region_bounded = region.intersection(envelope)
            vor_regions.append(region_bounded)

            if region_bounded.area != region.area:
                region_was_unbounded.append(1)
            else:
                region_was_unbounded.append(0)

            if region_bounded.area <= 0:
                raise RuntimeError(f"Unexpected: Polygon is completely outside map")

        return vor_regions, region_was_unbounded

    def clean_edges(self, edges: np.ndarray, discrete_pts: list, discrete_players: list, pt_to_poly: dict,
                    region_was_unbounded, pt_to_poly_idx):
        """Deletes invalid edges (polygons are not neighbors)

        Returns:
            list: List of valid edges
            list: List of player each edge belongs to. If edge connects enemies, then value is 4.
        """
        edge_player_id = []  # Player each edge belongs to
        edges_cleaned = []
        for idx, (p1_idx, p2_idx) in enumerate(edges):
            # TODO: Optimize this: We need to check if polygons are actually neighbors, because
            #  the scipy delaunay does not take into account the bounded polygons. Polys that
            #  meet at infinity will also count. Soln: Mirror all the points, so the bounds of the
            #  map will form natuarally, then delete the extra points and edges after.

            # test - We only do neighbor check on polygons that were modified. Faster.
            # Side-effect: Allows polys that may only share a corner to be classified as neighbors
            # Without this if, code is slower, but only true neighbors are filtered through.
            if (
                region_was_unbounded[pt_to_poly_idx[discrete_pts[p1_idx]]] == 1 or
                region_was_unbounded[pt_to_poly_idx[discrete_pts[p2_idx]]] == 1
            ):
                poly1 = pt_to_poly[discrete_pts[p1_idx]]
                poly2 = pt_to_poly[discrete_pts[p2_idx]]
                are_neighbors = self.poly_are_neighbors(poly1, poly2)
                if are_neighbors is None:
                    continue

            edges_cleaned.append((p1_idx, p2_idx))

            # Identify player edge belongs to
            player1 = discrete_players[p1_idx]  # Edges (from scipy) refer to points by their indices
            player2 = discrete_players[p2_idx]
            if player1 == player2:
                edge_player_id.append(player1)
            else:
                edge_player_id.append(4)

        return edges_cleaned, edge_player_id

        # def find_target_from_superpolygon(self, unit, superpolygon, s_neighbors, friendly_units, pt_to_poly_idx, vor_regions):
        #     neighboring_enemies = [n for n in s_neighbors if n not in friendly_units]
        #     neighboring_enemy_polygons = [pt_to_poly_idx[ne] for ne in neighboring_enemies]
        #
        #     candidates = set()
        #     for poly_idx in neighboring_enemy_polygons:
        #         polygon = vor_regions[poly_idx]
        #         intersection = superpolygon.intersection(polygon)
        #         if isinstance(intersection, shapely.geometry.LineString):
        #             points = intersection.coords
        #             for point in list(points):
        #                 candidates.add(point)
        #
        #     if len(candidates) == 0:
        #         target = unit  # if no enemies are found, stay in the same place
        #     else:
        #         target = max(list(candidates), key=lambda pt: (pt[0] - unit[0]) ** 2 + (pt[1] - unit[1]) ** 2)
        #
        #     return target


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
        self.current_day = 0

        self.home_offset = 0.5
        self.kdtree = None

        _MAP_W = self.max_dim
        self.spawn_loc = {
            0: (0, 0),
            1: (_MAP_W, 0),
            2: (_MAP_W, _MAP_W),
            3: (0, _MAP_W)
        }

    def get_incursions_polys(self, vor_regions, discrete_pt2player, poly_idx_to_pt):
        friendly_polygons = []
        for reg_idx, reg in enumerate(vor_regions):
            pt_ = poly_idx_to_pt[reg_idx]
            pl_ = discrete_pt2player[pt_]
            if pl_ == self.player_idx:
                friendly_polygons.append(reg)

        superpolygon = shapely.ops.unary_union(friendly_polygons)
        min_x, min_y, max_x, max_y = superpolygon.bounds
        sl = self.spawn_loc[self.player_idx]
        if sl == (min_x, min_y) or sl == (max_x, max_y):
            bound_points = shapely.geometry.MultiPoint([(max_x, min_y), (min_x, max_y)])
        else:
            bound_points = shapely.geometry.MultiPoint([(max_x, max_y), (min_x, min_y)])

        extended_polygon = bound_points.union(shapely.geometry.MultiPoint(superpolygon.exterior.coords))
        convexhull = extended_polygon.convex_hull
        incursions_ = convexhull.difference(superpolygon)

        if incursions_.is_empty:
            return []

        if isinstance(incursions_, shapely.geometry.MultiPolygon):
            incursions = list(incursions_.geoms)
        elif isinstance(incursions_, shapely.geometry.Polygon):
            incursions = [incursions_]
        else:
            incursions = []
            logging.warning(f"UNKNOWN condition: incursion is neither poly nor multipoly")

        viable_incursions = []
        edge_incursion_boundaries = []  # Outer edge near our border where incursion begins
        for incursion_ in incursions:
            # edges = shapely.geometry.LineString(incursion_.exterior.coords)
            sp = shapely.ops.shared_paths(convexhull.exterior, incursion_.exterior)
            if len(sp) == 0:
                continue
            edge_incursion_begin_f, edge_incursion_begin_b = sp
            edge_incursion_begin = edge_incursion_begin_f if not edge_incursion_begin_f.is_empty else edge_incursion_begin_b

            map_poly = shapely.geometry.Polygon([(0, 0), (0, 100), (100, 100), (100, 0), (0, 0)])
            valid_e = None
            for edge_ in edge_incursion_begin.geoms:
                if not map_poly.exterior.contains(edge_):
                    valid_e = edge_

            # edge_incursion_begin = edge_incursion_begin.geoms[0]  # There should be only 1 linestr in the multiplinestr
            if valid_e is not None and valid_e.length / incursion_.length <= 0.45:  # arbitrary number - consider anything that is at least square
                viable_incursions.append(incursion_)
                edge_incursion_boundaries.append(valid_e)

        # if len(edge_incursion_boundaries) > 0 and self.current_day > 193:
        #     plot_debug_incur(superpolygon, viable_incursions, edge_incursion_boundaries, self.current_day)

        return viable_incursions

    def get_groups_and_outliers(self, all_points, eps=3, min_samples=2, per_player=False):
        # DBSCAN - create dicts of groups and outliers
        groups_and_outliers_per_player = defaultdict(lambda: {'groups': defaultdict(list), 'outliers': []})
        all_groups_and_outliers = {'groups': defaultdict(list), 'outliers': []}

        if per_player:
            for pl in range(1, 5):
                if pl != self.player_idx:
                    np_points = np.array(all_points[pl])
                    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(np_points)
                    for i in range(len(np_points)):
                        group = clustering.labels_[i]
                        if group == -1:
                            groups_and_outliers_per_player[pl]['outliers'].append(np_points[i])
                        else:
                            groups_and_outliers_per_player[pl]['groups'][group].append(np_points[i])
            return groups_and_outliers_per_player
        else:
            np_all_points = []
            for pl in range(4):
                if pl != self.player_idx:
                    np_all_points += all_points[pl]
            np_all_points = np.array(np_all_points)
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(np_all_points)
            for i in range(len(np_all_points)):
                group = clustering.labels_[i]
                if group == -1:
                    all_groups_and_outliers['outliers'].append(np_all_points[i])
                else:
                    all_groups_and_outliers['groups'][group].append(np_all_points[i])
            return all_groups_and_outliers

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

        """TODO
        - Create a graph from all units (adj dict)
        - Create a tree from graph, root at our home base. 
            Use it to: build list of enemies at our border. list of friendlies at border.

        """
        map_size = self.max_dim

        _MAP_W = map_size
        spawn_loc = {
            0: (self.home_offset, self.home_offset),
            1: (_MAP_W - self.home_offset, self.home_offset),
            2: (_MAP_W - self.home_offset, _MAP_W - self.home_offset),
            3: (self.home_offset, _MAP_W - self.home_offset)
        }

        cg = CreateGraph(self.home_offset)
        discrete_pt2player, all_points = cg.create_pts_player_dict(unit_pos)

        # DBSCAN - create dicts of groups and outliers
        all_groups_and_outliers = self.get_groups_and_outliers(all_points, eps=3, min_samples=2, per_player=False)

        # Construct 2 lists: triangulation/voronoi takes discrete, strategy takes continuous position
        # Note: Discretized points will have duplicates, which are removed (disputed points, both removed).
        if len(all_points[self.player_idx]) == 0:
            return []  # No units on the map

        discrete_pts = list(discrete_pt2player.keys())
        discrete_players = list(discrete_pt2player.values())

        # Get polygon of Voronoi regions around each pt
        vor_regions, region_was_unbounded = cg.create_voronoi_regions(discrete_pts, map_size)

        # Find mapping from pts idx to polys (via nearest unit) and poly to player
        self.kdtree = scipy.spatial.KDTree(discrete_pts)
        pt_to_poly_idx, pt_to_poly, poly_idx_to_pt = cg.create_pt_to_poly_and_idx(self.kdtree, discrete_pts,
                                                                                  vor_regions)

        # Get the graph of connected pts via triangulation
        tri = scipy.spatial.Delaunay(np.array(discrete_pts))
        edges = cg.delaunay2edges(tri.simplices)  # Shape: [N, 2]
        # Clean edges
        edges, edge_player_id = cg.clean_edges(edges, discrete_pts, discrete_players, pt_to_poly,
                                               region_was_unbounded, pt_to_poly_idx)

        # if self.current_day > 1000:
        #     plot_units_and_edges(edges, edge_player_id, discrete_pts, discrete_players, pt_to_poly,
        #                          self.current_day)

        # Create adjacency list for graph of armies
        adj_dict = cg.create_adj_dict(edges, discrete_pts)

        moves = self.play_aggressive(all_points, pt_to_poly, adj_dict, discrete_pts)
        # if self.current_day <= (50 - self.spawn_days) or total_scores[self.player_idx] < max(total_scores):
        #     # Create union of all friendly polygons and list of its neighbors
        #     superpolygon, s_neighbors = self.create_superpolygon(vor_regions, pt_to_poly, adj_dict)
        #     moves = self.play_aggressive(vor_regions, pt_player_dict, pt_to_poly_idx, adj_dict, superpolygon,
        #                                  s_neighbors)
        # else:
        #     moves = self.play_cautious(unit_id, unit_pos, vor_regions, pt_player_dict, pt_to_poly_idx, adj_dict,
        #                                superpolygon, s_neighbors)

        # incursions = self.get_incursions_polys(vor_regions, discrete_pt2player, poly_idx_to_pt)
        # print(incursions)
        # if len(incursions) > 0:
        #     all_polys = list(pt_to_poly.values())
        #     plot_incursions(all_polys, incursions)

        self.current_day += 1
        return moves

    @staticmethod
    def move_toward_position(current, target):
        distance_to_target = np.sqrt((target[0] - current[0]) ** 2 + (target[1] - current[1]) ** 2)

        if distance_to_target == 0:
            angle_toward_target = 0.0
        else:
            angle_toward_target = np.arctan2(target[1] - current[1], target[0] - current[0])

        return max(1.0, distance_to_target), angle_toward_target

    # def play_cautious(self, unit_id, unit_pos, vor_regions, pt_player_dict, pt_to_poly_idx, adj_dict, superpolygon,
    #                   s_neighbors):
    #     moves = self.play_aggressive(vor_regions, pt_player_dict, pt_to_poly_idx, adj_dict, superpolygon, s_neighbors)
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

    def get_border_unit_target(self, unit, current_polygon, adj_dict, friendly_units_discrete, pt_to_poly):
        # Strategy - Move to furthest vertex of neighboring edges
        neighbors = adj_dict[unit]
        neighboring_enemies = [n for n in neighbors if n not in friendly_units_discrete]
        neighboring_enemy_polygons = [pt_to_poly[ne] for ne in neighboring_enemies]
        candidates = set()
        for polygon in neighboring_enemy_polygons:
            intersection = current_polygon.intersection(polygon)
            if (
                isinstance(intersection, shapely.geometry.LineString) or
                isinstance(intersection, shapely.geometry.Point)
            ):
                # NOTE: We allow polys that only share a corner to be considered as neighbors
                points = intersection.coords
                for point in list(points):
                    candidates.add(point)

        # further enemy polygon vertex on shared edges
        if len(candidates) > 0:
            return max(list(candidates), key=lambda pt: (pt[0] - unit[0]) ** 2 + (pt[1] - unit[1]) ** 2)
        else:
            return None
            # raise RuntimeError(f"Polygons do not share an edge. Could be they only share a corner, possibly"
            #                     f"error in cleaning edges from delaunay")

    def play_aggressive(self, all_points, pt_to_poly, adj_dict, discrete_pts):
        moves = []

        # Generate a move for every unit
        friendly_units = all_points[self.player_idx]
        friendly_units_discrete = [(int(x) + self.home_offset, int(y) + self.home_offset) for (x, y) in friendly_units]

        for unit_, unit in zip(friendly_units, friendly_units_discrete):
            if unit not in discrete_pts:
                continue

            # All polys, etc are indexed with discretized pts
            current_polygon = pt_to_poly[unit]

            neighbors = adj_dict[unit]
            neighboring_enemies = [n for n in neighbors if n not in friendly_units_discrete]
            neighboring_enemy_polygons = [pt_to_poly[ne] for ne in neighboring_enemies]

            if len(neighboring_enemies) == 0:
                new_neighboring_enemies = []
                for ne in neighbors:
                    neighboring_enemies_ne = [n for n in adj_dict[ne] if n not in friendly_units_discrete]
                    if len(neighboring_enemies_ne) > 0:
                        new_neighboring_enemies.append(ne)

                new_friendly_units_discrete = [u for u in friendly_units_discrete if u not in new_neighboring_enemies]

                target = self.get_border_unit_target(unit, current_polygon, adj_dict.copy(),
                                                     new_friendly_units_discrete, pt_to_poly)
                if target is None:
                    # Moving to centroid will spread units out
                    target = current_polygon.centroid.coords[0]
            else:
                target = self.get_border_unit_target(unit, current_polygon, adj_dict.copy(), friendly_units_discrete,
                                                     pt_to_poly)

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
            moves.append(self.move_toward_position(unit_, target))

        return moves
