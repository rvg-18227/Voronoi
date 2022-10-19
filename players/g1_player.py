import logging
import warnings
from typing import Tuple, Optional
from collections import defaultdict

import numpy as np
import shapely.errors
import shapely.geometry
import shapely.ops
import shapely.validation
import scipy
from sklearn.cluster import DBSCAN

# from tests import plot_funcs
# from tests.plot_funcs import plot_units_and_edges, plot_poly_list, plot_incursions, plot_line_list, plot_debug_incur, \
#     plot_dbscan

warnings.filterwarnings("ignore", category=shapely.errors.ShapelyDeprecationWarning)


class Unit:
    def __init__(self, pos: tuple, uid: int, player: int,
                 pos_int: tuple):
        self.pos = pos
        self.uid = uid
        self.player = player
        self.pos_int = pos_int

        self.role = None
        self.disputed = False
        self.neigh_fr = []
        self.neigh_ene = []
        self.poly = None
        self.poly_idx = None
        self.poly_vert = None
        self.move_cmd = (0, 0)  # angle, dist.
        self.target = None

    def reset(self):
        self.disputed = False
        self.neigh_fr = []
        self.neigh_ene = []
        self.poly = None
        self.poly_idx = None
        self.poly_vert = None
        self.move_cmd = (0, 0)  # angle, dist.
        self.target = None

    def nearest_fr(self):
        """Nearest friendly unit"""
        near_neigh = self.neigh_fr
        near_neigh = sorted(near_neigh, key=lambda x: dist_to_target(x, self.pos))
        if len(near_neigh) == 0:
            return None
        return near_neigh[0]

    def nearest_ene(self):
        """Nearest enemy unit"""
        near_neigh = self.neigh_ene
        near_neigh = sorted(near_neigh, key=lambda x: dist_to_target(x, self.pos))
        if len(near_neigh) == 0:
            return None
        return near_neigh[0]


def dist_to_target(unit: Unit, target: tuple[float, float]):
    current = np.array(unit.pos)
    target = np.array(target)
    dist_to_target = np.linalg.norm(target - current)
    return dist_to_target


def move_toward_position(unit: Unit, target: tuple[float, float]):
    current = unit.pos
    distance_to_target = min(1.0, dist_to_target(unit, target))

    if distance_to_target == 0:
        angle_toward_target = 0.0
    else:
        angle_toward_target = np.arctan2(target[1] - current[1], target[0] - current[0])

    return distance_to_target, angle_toward_target


class CommandoSquad:
    def __init__(self, units_list: list[Unit]):
        """Commando Squad designed to kill units
        Has a pincer motion: 3 units move cohesively, as they approach the target, the pin positions in front of the
        target and the left/right units flank the target and close in behind them, cutting them off from their support
        """
        self.max_units = 3
        if len(units_list) != self.max_units:
            logging.warning(f"Command squad only takes {self.max_units} units. Not using others")
            if len(units_list) > 2:
                units_list = units_list[:3]
            else:
                return

        self.units = []
        self.target: Optional[tuple[float, float]] = None
        self.target_reached = False
        self.dist_appr = 3.0  # how far pin will be from target
        self.target_unit = None  # A unit to kill
        # self.target_killed = False
        self.target_supp_vector = None  # The direction of nearest target's friendly unit

        self.add_units(units_list)
        self.pin = self.units[0]
        self.left = self.units[1]
        self.right = self.units[2]

        # Rotation matrices
        theta = np.deg2rad(-45)
        self.rot_left = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        theta = np.deg2rad(45)
        self.rot_right = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def set_target_unit(self, target_unit):
        self.target_unit = target_unit
        self.target = self.target_unit.pos
        # self.target_killed = False
        near_fr = self.target_unit.nearest_fr()
        if near_fr is not None:
            self.target_supp_vector = np.array([near_fr.pos[0] - self.target[0],
                                                near_fr.pos[1] - self.target[1]])
        else:
            self.target_supp_vector = None  # no nearest unit. Use unit pos

    def update_target(self, units_cls):
        """If a target unit is given, update the target point"""
        if self.target_unit is None:
            return

        self.target_killed = self.target_unit not in units_cls.values()
        if self.target_killed:
            logging.info(f"Target unit has been killed")
            self.target_unit = None
            self.target_supp_vector = None

            # If the target has been killed, retreat
            near_fr = self.pin.nearest_fr()
            if near_fr is not None:
                # if no other friendly unit nearby, doomed anyway
                self.target = near_fr.pos
        return

    def add_units(self, units_list):
        for unit in units_list:
            self.units.append(unit)
            unit.role = "cmdo_squad"

    def set_move_cmds(self):
        """If a target is present, move to that target to kill it"""
        if self.target_unit is not None:
            # TODO: More effecient: predict movement of target and move accordingly
            #    ref: https://stackoverflow.com/questions/17204513/how-to-find-the-interception-coordinates-of-a-moving-target-in-3d-space
            self.target = self.target_unit.pos

        if self.target is None:
            logging.warning(f"Commando squad does not have a target")
            return

        # If a target unit is given, get supp vector. Else, just use position info
        # Support vec is line joining target to it's nearest support
        if self.target_supp_vector is not None:
            supp_vec = self.target_supp_vector
        else:
            supp_vec = np.array([(self.target[0] - self.pin.pos[0]), (self.target[1] - self.pin.pos[1])])  # shape: [2,]
        supp_vec = (supp_vec / np.linalg.norm(supp_vec))  # normalize supp vec to magnitude 1

        vec_pin = -1 * supp_vec  # We want to approach from opposite direction
        vec_pin *= self.dist_appr  # stay this far from target
        self.pin.target = np.array(self.target) + vec_pin

        if dist_to_target(self.pin, self.pin.target) < 1:
            self.target_reached = True  # Set flag if near target

        factor_approach = 10  # Start fanning units out based on how near they are. Based on testing.
        if dist_to_target(self.pin, self.target) > self.dist_appr * factor_approach:
            # while far away, move as one
            self.left.target = self.pin.target
            self.right.target = self.pin.target
        else:
            vec_left = np.dot(self.rot_left, supp_vec)
            vec_right = np.dot(self.rot_right, supp_vec)
            self.left.target = self.target + vec_left * self.dist_appr
            self.right.target = self.target + vec_right * self.dist_appr

        moves = []
        for unit in self.units:
            # unit.target = self.target
            move = move_toward_position(unit, unit.target)
            unit.move_cmd = move
            moves.append(move)

        # for unit in [self.pin, self.left, self.right]:
        #     # unit.target = self.target
        #     move = move_toward_position(unit, unit.target)
        #     unit.move_cmd = move
        #     moves.append(move)

        return moves

    def disband_if_hurt(self):
        if len(self.units) < self.max_units:
            logging.warning(f"Commando squad has lost a unit. Disbanding squad.")
            self.disband_squad()

    def disband_squad(self):
        for unit in self.units:
            unit.role = None
            unit.target = None
        self.units = []

    def remove_killed_units(self, units_cls):
        self.units = [x for x in self.units if x in units_cls.values()]


class CautiousHeros:
    def __init__(self, player_idx, map_size):
        self.units = {
            "left": None,
            "right": None,
            "mid": None,
        }
        self.valid_units = []
        self.max_units = len(self.units)

        # Fixed positions around home base
        fort_positions_ = {
            0: [(1.6, 3.6), (3.6, 1.6), (1.5, 1.5)],
            1: [(1.6, map_size - 3.6), (3.6, map_size - 1.6), (1.5, map_size - 1.5)],
            2: [(map_size - 1.6, map_size - 3.6), (map_size - 3.6, map_size - 1.6), (map_size - 1.5, map_size - 1.5)],
            3: [(map_size - 1.6, 3.6), (map_size - 3.6, 1.6), (map_size - 1.5, 1.5)],
        }
        self.fort_positions = {
            "left": fort_positions_[player_idx][0],
            "right": fort_positions_[player_idx][1],
            "mid": fort_positions_[player_idx][2],
        }
        self.update()

    def update(self):
        self.valid_units = [x for x in self.units.values() if x is not None]

    def add_unit(self, unit):
        if unit in self.valid_units:
            logging.warning(f"Cautious heros already has this unit: {unit.uid}")
            return

        num_units = len(self.valid_units)
        if num_units >= self.max_units:
            logging.warning(f"Cautious Heros already has 2 units")
            return

        for key, val in self.units.items():
            if val is None:
                self.units[key] = unit
                unit.role = "cautious_heros"
                break
        self.update()

    def set_move_cmds(self):
        """If a target is present, move to that target to kill it"""
        moves = []
        for flank, unit in self.units.items():
            if unit is not None:
                unit.target = self.fort_positions[flank]
                move = move_toward_position(unit, unit.target)
                unit.move_cmd = move
                moves.append(move)
        return moves

    def remove_killed_units(self, units_cls: dict[int, Unit]):
        for flank, unit in self.units.items():
            if unit is not None and unit not in units_cls.values():
                self.units[flank] = None
        self.update()


class CreateGraph:
    def __init__(self, home_offset, player_idx):
        self.home_offset = home_offset
        self.player_idx = player_idx
        self.kdtree = None
        self.units_cls = {}

    @staticmethod
    def get_unique_uid(pl, uid):
        """Create a unique id for each individual unit. Simulator generates unique unit ids per player"""
        return uid + 10000 * pl

    def print_roles(self):
        # debug func
        a = [x.role for x in self.units_cls.values() if x.player == self.player_idx]
        print(a)

    def create_pts_idx_dict(self, unit_pos, unit_id) -> tuple[dict[tuple, int], dict[int, Unit]]:
        """Create list of all points converted to our Unit class data structures and lists indexing into it
        Secondary lists let access certain points by their ID, such as all unique discrete points

        Note: Unit ID is not unique between players. Multiple players can have unit with the same ID.

        Returns:
            pt_to_uuid (dict): {pos_int: uuid}
            units_cls (dict): {uuid: Unit}
        """
        discretept_to_uuid = {}  # discrete pts to player
        disputed_pts = []
        units_cls = {}
        for pl in range(4):
            for spt, uid_ in zip(unit_pos[pl], unit_id[pl]):
                # Quantize unit pos to cell. We assume cell origin at center.
                x, y = spt.coords[0]
                pos_int = (int(x) + self.home_offset, int(y) + self.home_offset)

                uuid = self.get_unique_uid(pl, uid_)
                discretept_to_uuid[pos_int] = uuid

                # Update existing unit object instead of creating new unit object
                if uuid in self.units_cls:
                    unit = self.units_cls[uuid]
                    unit.reset()  # Reset all the other attrb like neighbors. Will be re-calculated
                    unit.pos = (x, y)
                    unit.pos_int = pos_int
                    units_cls[uuid] = self.units_cls[uuid]
                else:
                    unit_c = Unit((x, y), uuid, pl, pos_int)
                    units_cls[uuid] = unit_c

                # Identify disputed units - will be removed so that they don't affect voronoi regions
                if pos_int in discretept_to_uuid:
                    uuid = discretept_to_uuid[pos_int]
                    player_existing = units_cls[uuid].player
                    if player_existing != pl:
                        # Disputed cell - remove/mark all later
                        # pl, uid required to access unit cls list
                        disputed_pts.append((pos_int, uuid))

        # Disputed cell units are removed from discrete list so that they don't affect voronoi regions
        for pos_int, uid in disputed_pts:
            if pos_int in discretept_to_uuid:
                discretept_to_uuid.pop(pos_int)
            units_cls[uid].disputed = True

        self.units_cls = units_cls
        return discretept_to_uuid, units_cls

    def discretize_unit(self, unit_):
        unit = (int(unit_[0]) + self.home_offset, int(unit_[1]) + self.home_offset)
        return unit

    def set_unit_polys(self, units_cls: dict[int, Unit],
                       vor_regions: list[shapely.geometry.Polygon]) -> dict[int, Unit]:
        """Assign a polygon to each unit"""
        # Get points from the unique unit ids
        discrete_pts = list({x.pos_int for x in units_cls.values()})
        # kdtree contains all the discrete points
        self.kdtree = scipy.spatial.KDTree(discrete_pts)

        # Mark each unit with it's polygon
        discrete_pt_to_poly = {}
        for region_idx, region in enumerate(vor_regions):
            # Voronoi regions are all convex. Nearest pt to centroid must be point belonging to region
            centroid = region.centroid.coords[0]
            _, ii = self.kdtree.query(centroid, k=1)  # index of nearest pt

            discrete_pt_to_poly[discrete_pts[ii]] = (region_idx, region)

        # For each point, set the corresponding polygon
        for unit in units_cls.values():
            unit.poly_idx = discrete_pt_to_poly[unit.pos_int][0]
            unit.poly = discrete_pt_to_poly[unit.pos_int][1]
            unit.poly_vert = list(unit.poly.exterior.coords)

        return units_cls

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
    def set_unit_neighbors(edges, units_cls) -> dict[int, Unit]:
        uuids = list(units_cls.keys())
        for val in edges:
            p1_idx, p2_idx = val
            u1 = units_cls[uuids[p1_idx]]
            u2 = units_cls[uuids[p2_idx]]

            # modify unit classes dict
            if u1.player == u2.player:
                u1.neigh_fr.append(u2)
                u2.neigh_fr.append(u1)
            else:
                u1.neigh_ene.append(u2)
                u2.neigh_ene.append(u1)
        return units_cls

    @staticmethod
    def get_valid_polygon(region):
        # Sometimes the region returned by voronoi is invalid. Example: A poly with a line attached,
        # causing self-intersection. We fix this using shapely's validation module.
        multip = shapely.validation.make_valid(region)
        for obj in multip:
            if isinstance(obj, shapely.geometry.Polygon):
                # Get only a polygon (ignore linestr, etc) from this erroneous shape
                region = obj
        if not isinstance(region, shapely.geometry.Polygon):
            return None
        return region

    def create_voronoi_regions(self, units_cls: dict[int, Unit], map_size: int):
        """Get polygon of Voronoi regions around each discretised pt (it's cell origin).
        The polygons are bounded to the limits of the map manually.

        Args:
            units_cls: Mapping of uuids to unit objects
            map_size: Size of map

        Returns:
            list: List of polygons (not in same order as input points)
            list: List of ints. 1 if corresponding poly was corrected (was out of bounds), 0 otherwise
        """
        # Get points from the unique unit ids
        # voronoi throws an error if near-duplicate points present. So we use the origin of the cell for each point
        # This means that if multiple points are within the same cell, they will all map to the same poly
        # disputed points do not contribute to the voronoi regions
        pts = list({x.pos_int for x in units_cls.values() if not x.disputed})

        _points = shapely.geometry.MultiPoint(pts)
        envelope = shapely.geometry.box(0, 0, map_size, map_size)
        vor_regions_ = shapely.ops.voronoi_diagram(_points, edges=False)
        vor_regions_ = list(vor_regions_)  # Convert to a list of Polygon

        # The polys aren't bounded. Fix manually. Mark which polys were not bounded (goes outside map)
        vor_regions = []
        region_was_unbounded = []  # Whether a region was out of bounds
        for region in vor_regions_:
            if not region.is_valid:
                region = self.get_valid_polygon(region)
                if region is None:
                    logging.warning(f"Got invalid polygon from voronoi. Could not fix")
                    continue

            region_bounded = envelope.intersection(region)
            if region_bounded.area <= 0:
                logging.warning(f"Unexpected: Polygon is completely outside map")
                continue

            # Can be used for more efficient validating of edges.
            # But this allows polys with only a shared corner to be set as neighbors
            # NOTE: Allowing polys with only corners to be considered neighbors will results in errors in d1/d2 units
            #   not finding neighboring enemies
            vor_regions.append(region_bounded)
            if region_bounded.area != region.area:
                region_was_unbounded.append(1)
            else:
                region_was_unbounded.append(0)

        return vor_regions, region_was_unbounded

    def get_delaunay_edges(self, units_cls: dict[int, Unit]) -> tuple[list[tuple[int, int]], list[int]]:
        """Get the edges between units as per Delaunay Triangulation.
        Deletes invalid edges (bounded polygons are not neighbors)

        Returns:
            list: List of valid edges: [(p1_idx, p2_idx)]
            list: List of owners of each edge. 0-3: same player unit on both ends of edge,
                4: diff players on either end of the edge
        """
        # Cautious heros can mess with the d2 defense logic, because d1 will have an edge to them and not to
        #   newly spawned units in the home base
        pts = [x.pos for x in units_cls.values()]
        units_c = list(units_cls.values())
        tri = scipy.spatial.Delaunay(np.array(pts))
        edges = self.delaunay2edges(tri.simplices)

        # Remove invalid edges - edge exists between unbounded polys, but not bounded polys.
        edges_cleaned = []
        edge_player_id = []
        for idx, (p1_idx, p2_idx) in enumerate(edges):
            poly1 = units_c[p1_idx].poly
            poly2 = units_c[p2_idx].poly
            are_neighbors = self.poly_are_neighbors(poly1, poly2)
            if are_neighbors is None:
                continue

            edges_cleaned.append((p1_idx, p2_idx))

            # Identify player edge belongs to
            player1 = units_c[p1_idx].player  # Edges (from scipy) refer to points by their indices
            player2 = units_c[p2_idx].player
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
        self.current_day = -1

        self.home_offset = 0.5
        self.kdtree = None

        _MAP_W = self.max_dim
        self.spawn_loc = {
            0: (0, 0),
            1: (_MAP_W, 0),
            2: (_MAP_W, _MAP_W),
            3: (0, _MAP_W)
        }

        self.cg = CreateGraph(self.home_offset, self.player_idx)
        self.cautious_heros = CautiousHeros(self.player_idx, _MAP_W)  # list of uuids
        self.commando_squads = []

        self.units_to_be_drafted = []
        self.draft_cmdo_start = False
        self.num_commandos = 3

    def incursions_to_enemies(self, incursions, unit_cls):
        in_poly_list = [[] for _ in range(len(incursions))]
        for unit in unit_cls.values():
            if unit.player != self.player_idx:
                for i in range(len(incursions)):
                    if incursions[i].contains(shapely.geometry.Point(unit.pos)):
                        in_poly_list[i].append(unit)
                        break
        return in_poly_list

    def get_incursions_polys(self, units_cls: dict[int, Unit]):
        # todo - get list of friendly units and get their polygons. No need for poly_idx_to_pt
        friendly_units = [x for x in units_cls.values() if x.player == self.player_idx]
        friendly_polygons = [x.poly for x in friendly_units]

        superpolygon = shapely.ops.unary_union(friendly_polygons)
        min_x, min_y, max_x, max_y = superpolygon.bounds
        sl = self.spawn_loc[self.player_idx]
        if sl == (min_x, min_y) or sl == (max_x, max_y):
            bound_points = shapely.geometry.MultiPoint([(max_x, min_y), (min_x, max_y)])
        else:
            bound_points = shapely.geometry.MultiPoint([(max_x, max_y), (min_x, min_y)])

        if superpolygon.geom_type == 'MultiPolygon':
            superpolygon = superpolygon.geoms[0]

        extended_polygon = bound_points.union(shapely.geometry.MultiPoint(superpolygon.exterior.coords))
        convexhull = extended_polygon.convex_hull
        incursions_ = convexhull.difference(superpolygon)

        if incursions_.is_empty:
            return [], []

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
                    break

            # edge_incursion_begin = edge_incursion_begin.geoms[0]  # There should be only 1 linestr in the multiplinestr
            if valid_e is not None and valid_e.length / incursion_.length <= 0.45:  # arbitrary number - consider anything that is at least square
                viable_incursions.append(incursion_)
                edge_incursion_boundaries.append(valid_e)

        # if len(edge_incursion_boundaries) > 0 and self.current_day > 193:
        #     plot_debug_incur(superpolygon, viable_incursions, edge_incursion_boundaries, self.current_day)

        return viable_incursions, edge_incursion_boundaries

    def get_groups_and_outliers(self, all_points, eps=3, min_samples=2, per_player=False):
        # DBSCAN - create dicts of groups and outliers
        groups_and_outliers_per_player = defaultdict(lambda: {'groups': defaultdict(list), 'outliers': []})
        all_groups_and_outliers = {'groups': defaultdict(list), 'outliers': []}

        if per_player:
            for pl in range(4):
                if pl != self.player_idx and len(all_points[pl]) > 0:
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
            np_all_points = np.empty((0, 2), dtype=float)
            for pl in range(4):
                if pl != self.player_idx and len(all_points[pl]) > 0:
                    np_points = np.array(all_points[pl])
                    np_all_points = np.append(np_all_points, np_points)
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
        unit_id_ = []
        for player in range(4):
            x = [int(x) for x in unit_id[player]]
            unit_id_.append(x)
        unit_id = unit_id_

        self.current_day += 1
        if len(unit_pos[self.player_idx]) == 0:
            return []  # No units on the map

        # All other lists (discrete pts, etc) index into list of Units class.
        discretept_to_uuid, units_cls = self.cg.create_pts_idx_dict(unit_pos, unit_id)

        # Discrete are accessed like this if needed
        # discrete_uuids = list(discretept_to_uuid.values())
        # discrete_pts = [units_cls[x].pos_int for x in discrete_uuids]

        # DBSCAN - create dicts of groups and outliers
        # all_groups_and_outliers = self.get_groups_and_outliers(all_points, eps=3, min_samples=2, per_player=False)
        # if self.current_day % 20 == 0:
        #     plot_dbscan(player_groups_and_outliers, self.current_day)

        # Get polygon of Voronoi regions around each pt (discrete)
        map_size = self.max_dim
        vor_regions, region_was_unbounded = self.cg.create_voronoi_regions(units_cls, map_size)

        # Assign mapping from units to polys
        units_cls = self.cg.set_unit_polys(units_cls, vor_regions)

        # Graph of units - set neighboring friendlies/enemies
        edges, edge_player_id = self.cg.get_delaunay_edges(units_cls)
        units_cls = self.cg.set_unit_neighbors(edges, units_cls)

        # if self.current_day > 78:
        #     plot_units_and_edges(edges, edge_player_id, units_cls, self.current_day)

        # INCURSIONS - Identify regions where enemies are encroaching into our territory.
        incursions, edge_incursion_boundaries = self.get_incursions_polys(units_cls)
        units_in_incursions = self.incursions_to_enemies(incursions, units_cls)

        ideal_incur_units = []  # one unit per incursion
        for units_in_inc, edge_incur in zip(units_in_incursions, edge_incursion_boundaries):
            dist = 0
            unit_max = None
            for unit in units_in_inc:
                d = shapely.geometry.Point(unit.pos).distance(edge_incur)
                if d > dist:
                    dist = d
                    unit_max = unit
            ideal_incur_units.append(unit_max)

        # print(incursions)
        # if len(incursions) > 0 and self.current_day % 20 == 0:
        #     all_polys = [x.poly for x in units_cls.values()]
        #     plot_incursions(all_polys, incursions, units_in_incursions, self.current_day)

        # Analyze map
        # d1 = [] -> strat_border_patrol()
        # d2 = [] -> strat_border_patrol()
        # defenders = [] -> strat_centroid()
        # cautious = [] -> strat_cautious()
        # strat_move_to_edge_center
        # strat_move_to_mean_enemy_neighbors
        # a = [x.role for x in units_cls.values() if x.player == self.player_idx]
        avail_units = {x for x in units_cls.values() if x.player == self.player_idx}

        # # cautious heros
        # self.cautious_heros.remove_killed_units(units_cls)  # Remove killed units
        # for unit in self.cautious_heros.valid_units:
        #     avail_units.remove(unit)  # remove from list of avail units
        # if self.current_day >= 50:
        #     # Whenever new unit is spawned, add it to cautious heros if needed
        #     # todo - if a cautious hero is killed, new unit will not be added until next spawn day.
        #     #   Should we draft forcefully from defense units? We can defend coordinated atck from default player.
        #     if (
        #         self.current_day % self.spawn_days == 0
        #         and len(self.cautious_heros.valid_units) < self.cautious_heros.max_units
        #     ):
        #         latest_unit_id = sorted([x.uid for x in avail_units])[-1]
        #         unit = units_cls[latest_unit_id]
        #         self.cautious_heros.add_unit(unit)
        #         avail_units.remove(unit)

        # commando squads
        # draft 3 units
        # todo - logic for drafting units into cmdo squads
        if not self.draft_cmdo_start:
            self.draft_cmdo_start = True
            self.units_to_be_drafted = []

        total_com = max(self.num_commandos, len(ideal_incur_units))
        if self.current_day > 50 and len(self.commando_squads) < total_com and self.draft_cmdo_start:
            # if self.current_day > 50 and len(self.commando_squads) < len(ideal_incur_units) and self.draft_cmdo_start:
            if len(self.units_to_be_drafted) < 3:
                if self.current_day % self.spawn_days == 0:
                    latest_unit_id = sorted([x.uid for x in avail_units])[-1]
                    unit = units_cls[latest_unit_id]
                    self.units_to_be_drafted.append(unit)
            else:
                csq = CommandoSquad(self.units_to_be_drafted)
                self.commando_squads.append(csq)
                self.draft_cmdo_start = False

        for unit in self.units_to_be_drafted:
            if unit in avail_units:
                avail_units.remove(unit)

        valid_squads = []
        for csq in self.commando_squads:
            csq.remove_killed_units(units_cls)
            csq.disband_if_hurt()  # commando squad cannot perform with less than 3 units
            for unit in csq.units:
                if unit in avail_units:
                    avail_units.remove(unit)  # remove from list of avail units
            if len(csq.units) > 0:
                valid_squads.append(csq)
        self.commando_squads = valid_squads

        # d1 - highest priority - layer 1 defense
        d1_units = {x for x in avail_units if len(x.neigh_ene) > 0}
        for unit in d1_units:
            avail_units.remove(unit)
            unit.role = "d1"

        # d2 - layer 2 defense - all of d1's friendly neighbors not already in d1
        d2_units = set()
        for unit in d1_units:
            for nf in unit.neigh_fr:
                if nf not in d1_units and nf in avail_units:
                    nf.role = "d2"
                    d2_units.add(nf)
                    avail_units.remove(nf)

        # defenders - internal units spread within area of control - centroid strat
        def_units = set()
        for unit in avail_units:
            def_units.add(unit)
            unit.role = "def"
        for unit in def_units:
            avail_units.remove(unit)

        # Set the move_cmd attr for each unit
        # _ = self.cautious_heros.set_move_cmds()  # Defensive ring
        _ = self.strat_border_patrol(d1_units)
        _ = self.strat_border_patrol(d2_units, d1_units)
        _ = self.strat_move_to_centroid(def_units)

        """Commando squad 
                - form a Commando squad of 3 units
                - identify incursions
                - find enemy units in incursions
                - filter enemy units based on their isolations (dbscan)
                - surround and kill enemy target
        """

        # COMMANDO =- Assign units to kill

        # ene_units_border = set()
        # for unit in d1_units:
        #     for ene in unit.neigh_ene:
        #         ene_units_border.add(ene)
        # ene_units_border = list(ene_units_border)

        last_c = 0
        for idx, (csq, ideal_incur_u) in enumerate(zip(self.commando_squads, ideal_incur_units)):
            csq.update_target(units_cls)
            tar_unit = ideal_incur_u
            print(f"target selection: {csq}, {tar_unit.uid}")
            csq.set_target_unit(tar_unit)
            csq.update_target(units_cls)
            _ = csq.set_move_cmds()
            last_c = idx

        if last_c < len(self.commando_squads):
            all_enemies = [x for x in units_cls.values() if x.player != self.player_idx]
            for idx in range(last_c + 1, len(self.commando_squads)):
                csq = self.commando_squads[idx]
                csq.update_target(units_cls)
                all_enemies.sort(key=lambda x: dist_to_target(csq.pin, x.pos))

                tar_unit = all_enemies[0]
                all_enemies.remove(all_enemies[0])

                print(f"target selection: {csq}, {tar_unit.uid}")
                csq.set_target_unit(tar_unit)
                csq.update_target(units_cls)
                _ = csq.set_move_cmds()

        for idx, csq in enumerate(self.commando_squads):
            try:
                tar = csq.target_unit.uid
            except AttributeError:
                tar = None
            logging.info(f"Cmdo {idx} target: {tar} at {csq.target}")
            for unit in csq.units:
                logging.info(f"  moves: {unit.move_cmd}, curr_pos: {unit.pos}")

        # accumulate all moves in correct order, and return
        moves = []
        for uid in unit_id[self.player_idx]:
            unit = units_cls[self.cg.get_unique_uid(self.player_idx, uid)]
            moves.append(unit.move_cmd)

        return moves

    def strat_border_patrol(self,
                            units_list: set[Unit],
                            valid_friendlies: set[Unit] = None) -> list[tuple]:
        """Move unit to furthest of all vertices of polygon associated with neighboring enemies
        We pass in list of neigh enemies to generalize depth 1 and depth 2 (depth 2 is 2 layers from border)

        Args:
            units_list: List of units for which to generate move cmds
            valid_friendlies: If given, D2/D3 strat will be used. D2 units looks at neighboring D1 units (friendlies)

        Returns:
            list: list of movement cmds for each unit
        """
        moves = []
        for unit in units_list:
            if valid_friendlies is not None:
                # D2/D3 strat
                neighboring_enemies = [n for n in unit.neigh_fr if n in valid_friendlies]
            else:
                # D1 strat
                neighboring_enemies = [n for n in unit.neigh_ene]

            neighboring_enemy_polygons = [n.poly for n in neighboring_enemies]
            current_polygon = unit.poly

            candidates = set()
            for polygon in neighboring_enemy_polygons:
                intersection = current_polygon.intersection(polygon)
                if (
                    isinstance(intersection, shapely.geometry.LineString)
                    # or isinstance(intersection, shapely.geometry.Point)  # NOTE: We used to allow polys that only share a corner to be considered as neighbors
                ):
                    points = intersection.coords
                    for point in list(points):
                        candidates.add(point)

            # further enemy polygon vertex on shared edges
            if len(candidates) > 0:
                target = max(list(candidates),
                             key=lambda pt: (pt[0] - unit.pos[0]) ** 2 + (pt[1] - unit.pos[1]) ** 2)
                move = move_toward_position(unit, target)
                unit.target = target
            else:
                logging.error(f"Unit Doesn't have valid neighboring enemies polygon (no common vertices)")
                move = (0, 0)

            unit.move_cmd = move
            moves.append(move)

        return moves

    def strat_move_to_centroid(self, units_list: set[Unit]) -> list[tuple]:
        """Defensive units (center of region) - Move to centroid of unit's voronoi polygon
        Causes units to naturally spread out in the area, as voronoi cells try to minimize their
        surface area (more circular shape). However, units will be more dense near the origin
        """
        moves = []
        for unit in units_list:
            target = unit.poly.centroid.coords[0]
            move = move_toward_position(unit, target)
            unit.move_cmd = move
            unit.target = target
            moves.append(move)
        return moves

    def strat_move_to_edge_center(
        self, units_list: set[Unit], valid_friendlies: set[Unit] = None
    ) -> list[tuple]:
        """Strategy - Move to middle of closest edge
        Offensive - Units near the border go to the middle of nearest edge of neighboring enemy unit's polygons
        This is more even than the furthest vertex of the polygon, as every unit will have one corresponding
        edge to move to.
        """
        moves = []
        for unit in units_list:
            neighboring_enemies = [n for n in unit.neigh_ene]
            neighboring_enemy_polygons = [n.poly for n in neighboring_enemies]
            current_polygon = unit.poly

            candidates = set()
            for polygon in neighboring_enemy_polygons:
                intersection = current_polygon.intersection(polygon)
                if isinstance(intersection, shapely.geometry.LineString):
                    points = intersection.coords[:]  # tuple
                    points_np = np.array(points)
                    edge_center = tuple(points_np.mean(axis=0).tolist())
                    candidates.add(edge_center)

            min_dist = float('inf')
            target = None
            for edge_c in candidates:
                dist = np.linalg.norm(np.array(edge_c) - np.array(unit.pos))
                if dist < min_dist:
                    min_dist = dist
                    target = edge_c

            move = move_toward_position(unit, target)
            unit.move_cmd = move
            unit.target = target
            moves.append(move)

        return moves

    def strat_move_to_mean_enemy_neighbors(
        self, units_list: set[Unit], valid_friendlies: set[Unit] = None
    ) -> list[tuple]:
        """Strategy - Move to mean of neighboring enemies
        Offensive strategy - Move towards the mean of the neighboring enemy units.
        Causes our units to advance. Generally suicidal against dense formations.
        """
        moves = []
        for unit in units_list:
            neighboring_enemies = [n for n in unit.neigh_ene]
            neig_ene = np.array([x.pos for x in neighboring_enemies])  # (N, 2)
            neig_ene_mean = neig_ene.mean(axis=0)
            target = tuple(neig_ene_mean.tolist())

            move = move_toward_position(unit, target)
            unit.move_cmd = move
            unit.target = target
            moves.append(move)
        return moves

