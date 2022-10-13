import cv2
import logging
import os
import pickle
import time
import scipy
import signal
import numpy as np
from shapely.geometry import Point
from remi import start
from voronoi_app import VoronoiApp
import constants
from utils import *
from players.default_player import Player as DefaultPlayer
from players.g1_player import Player as G1_Player
from players.g2_player import Player as G2_Player
from players.g3_player import Player as G3_Player
from players.g4_player import Player as G4_Player
from players.g5_player import Player as G5_Player
from players.g6_player import Player as G6_Player
from players.g7_player import Player as G7_Player
from players.g8_player import Player as G8_Player
from players.g9_player import Player as G9_Player


class VoronoiGame:
    def __init__(self, player_list, args):
        self.start_time = time.time()
        self.end_time = None
        self.voronoi_app = None
        self.use_gui = not args.no_gui
        self.do_logging = not args.disable_logging
        self.use_timeout = not args.disable_timeout

        self.logger = logging.getLogger(__name__)
        # create file handler which logs even debug messages
        if self.do_logging:
            self.logger.setLevel(logging.DEBUG)
            self.log_dir = args.log_path
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(self.log_dir, 'debug.log'), mode="w")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter('%(message)s'))
            fh.addFilter(MainLoggingFilter(__name__))
            self.logger.addHandler(fh)
            result_path = os.path.join(self.log_dir, "results.log")
            rfh = logging.FileHandler(result_path, mode="w")
            rfh.setLevel(logging.INFO)
            rfh.setFormatter(logging.Formatter('%(message)s'))
            rfh.addFilter(MainLoggingFilter(__name__))
            self.logger.addHandler(rfh)
        else:
            if args.log_path:
                self.logger.setLevel(logging.INFO)
                result_path = args.log_path
                self.log_dir = os.path.dirname(result_path)
                if self.log_dir:
                    os.makedirs(self.log_dir, exist_ok=True)
                rfh = logging.FileHandler(result_path, mode="w")
                rfh.setLevel(logging.INFO)
                rfh.setFormatter(logging.Formatter('%(message)s'))
                rfh.addFilter(MainLoggingFilter(__name__))
                self.logger.addHandler(rfh)
            else:
                self.logger.setLevel(logging.ERROR)
                self.logger.disabled = True

        if args.seed == 0:
            args.seed = None
            self.logger.info("Initialise random number generator with no seed")
        else:
            self.logger.info("Initialise random number generator with seed {}".format(args.seed))

        self.rng = np.random.default_rng(args.seed)

        self.spawn_day = args.spawn
        self.last_day = args.last
        self.base = []
        for i in range(constants.no_of_players):
            self.base.append(Point(constants.base[i]))

        self.fast_map = FastMapState(constants.max_map_dim, constants.base)

        self.players = []
        self.player_names = []

        self.map_states = [[[[0 for z in range(constants.max_map_dim)] for k in range(constants.max_map_dim)] for j in
                            range(constants.day_states)] for i in range(self.last_day)]

        self.player_score = [[[0 for k in range(constants.no_of_players)] for j in range(constants.day_states)] for i in
                             range(self.last_day)]
        self.player_total_score = [[0 for j in range(constants.no_of_players)] for i in range(self.last_day)]

        self.unit_id = [[[[] for k in range(constants.no_of_players)] for j in range(constants.day_states)] for i
                        in range(self.last_day)]
        self.unit_pos = [[[[] for k in range(constants.no_of_players)] for j in range(constants.day_states)] for i
                         in range(self.last_day)]

        self.home_path = [[[[False for z in range(constants.max_map_dim)] for k in range(constants.max_map_dim)] for j
                           in range(constants.no_of_players)] for i in range(self.last_day)]

        self.end_message_printed = False

        self.add_players(player_list)

        if self.use_gui and args.web_gui:
            # Use the web gui
            config = dict()
            config["address"] = args.address
            config["start_browser"] = not args.no_browser
            config["update_interval"] = 0.5
            config["userdata"] = (self, self.logger)
            if args.port != -1:
                config["port"] = args.port
            start(VoronoiApp, **config)

        self.dump_state = args.dump_state

    def add_players(self, player_list):
        player_count = dict()
        for player_name in player_list:
            if player_name not in player_count:
                player_count[player_name] = 0
            player_count[player_name] += 1

        count_used = {k: 0 for k in player_count}

        i = 0
        for player_name in player_list:
            if player_name in constants.possible_players:
                if player_name.lower() == "d":
                    player_class = DefaultPlayer
                    base_player_name = "Default Player"
                else:
                    player_class = eval("G{}_Player".format(player_name))
                    base_player_name = "Group {}".format(player_name)

                count_used[player_name] += 1
                if player_count[player_name] == 1:
                    self.add_player(player_class, "{}".format(base_player_name), base_player_name=base_player_name,
                                    idx=i)
                else:
                    self.add_player(player_class, "{}.{}".format(base_player_name, count_used[player_name]),
                                    base_player_name=base_player_name, idx=i)
            else:
                self.logger.error("Failed to insert player {} since invalid player name provided.".format(player_name))

            i += 1

    def add_player(self, player_class, player_name, base_player_name, idx):
        if player_name not in self.player_names:
            self.logger.info(
                "Adding player {} from class {}".format(player_name, player_class.__module__))
            precomp_dir = os.path.join("precomp", base_player_name)
            os.makedirs(precomp_dir, exist_ok=True)

            start_time = 0
            is_timeout = False
            if self.use_timeout:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(constants.timeout)
            try:
                start_time = time.time()
                player = player_class(rng=self.rng, logger=self.get_player_logger(player_name),
                                      total_days=self.last_day, spawn_days=self.spawn_day, player_idx=idx,
                                      spawn_point=self.base[idx], min_dim=constants.min_map_dim,
                                      max_dim=constants.max_map_dim, precomp_dir=precomp_dir)
                if self.use_timeout:
                    signal.alarm(0)  # Clear alarm
            except TimeoutException:
                is_timeout = True
                player = None
                self.logger.error(
                    "Initialization Timeout {} since {:.3f}s reached.".format(player_name, constants.timeout))

            init_time = time.time() - start_time

            if not is_timeout:
                self.logger.info("Initializing player {} took {:.3f}s".format(player_name, init_time))
            self.players.append(player)
            self.player_names.append(player_name)
        else:
            self.logger.error("Failed to insert player as another player with name {} exists.".format(player_name))

    def get_player_logger(self, player_name):
        player_logger = logging.getLogger("{}.{}".format(__name__, player_name))

        if self.do_logging:
            player_logger.setLevel(logging.INFO)
            # add handler to self.logger with filtering
            player_fh = logging.FileHandler(os.path.join(self.log_dir, '{}.log'.format(player_name)), mode="w")
            player_fh.setLevel(logging.DEBUG)
            player_fh.setFormatter(logging.Formatter('%(message)s'))
            player_fh.addFilter(PlayerLoggingFilter(player_name))
            self.logger.addHandler(player_fh)
        else:
            player_logger.setLevel(logging.ERROR)
            player_logger.disabled = True

        return player_logger

    def play_game(self):
        for i in range(self.last_day):
            self.play_day(i)

        self.print_results()

    def print_results(self):
        self.end_time = time.time()
        result = self.get_state(self.last_day - 1, 2)
        print("\nDay {} - {}".format(result["day"], result["day_states"]))
        print("\nPlayers - {}".format(result["player_names"]))
        print("Day Score - {}".format(result["player_score"]))
        print("Total Score - {}".format(result["player_total_score"]))
        print("{} Units - {}".format(result["player_names"][0], result["unit_id"][0]))
        print("{} Unit Positions - {}".format(result["player_names"][0],
                                              [(float(i.x), float(i.y)) for i in result["unit_pos"][0]]))
        print("{} Units - {}".format(result["player_names"][1], result["unit_id"][1]))
        print("{} Unit Positions - {}".format(result["player_names"][1],
                                              [(float(i.x), float(i.y)) for i in result["unit_pos"][1]]))
        print("{} Units - {}".format(result["player_names"][2], result["unit_id"][2]))
        print("{} Unit Positions - {}".format(result["player_names"][2],
                                              [(float(i.x), float(i.y)) for i in result["unit_pos"][2]]))
        print("{} Units - {}".format(result["player_names"][3], result["unit_id"][3]))
        print("{} Unit Positions - {}".format(result["player_names"][3],
                                              [(float(i.x), float(i.y)) for i in result["unit_pos"][3]]))
        print("\nTime Elapsed - {:.3f}s".format(self.end_time - self.start_time))

        if self.dump_state:
            with open("game.pkl", "wb+") as f:
                pickle.dump(
                    {
                        "map_states": self.map_states,
                        "player_names": self.player_names,
                        "player_score": self.player_score,
                        "player_total_score": self.player_total_score,
                        "unit_id": self.unit_id,
                        "unit_pos": self.unit_pos,
                        "home_path": self.home_path,
                        "last_day": self.last_day,
                        "spawn_day": self.spawn_day
                    },
                    f,
                )

    def play_day(self, day):
        if day != 0:
            for i in range(constants.no_of_players):
                for j in range(len(self.unit_id[day - 1][2][i])):
                    self.unit_id[day][0][i].append(self.unit_id[day - 1][2][i][j])
                    self.unit_pos[day][0][i].append(self.unit_pos[day - 1][2][i][j])

        if day % self.spawn_day == 0:
            # new unit spawned. Cannot copy prev day scores. Re-calculate the scores.
            self.spawn_new(day, str((day // self.spawn_day) + 1))
            score, map_state = self.fast_map.update_map_state(day, 0, self.unit_pos)
            self.player_score[day][0] = score
            self.map_states[day][0] = map_state
        else:
            # copy prev day's end state and score to this day's init state and score
            for i in range(constants.max_map_dim):
                for j in range(constants.max_map_dim):
                    self.map_states[day][0][i][j] = self.map_states[day - 1][2][i][j]

            for i in range(constants.no_of_players):
                self.player_score[day][0][i] = self.player_score[day - 1][2][i]

        returned_action = None
        for i in range(constants.no_of_players):
            if len(self.unit_id[day][0][i]) > 0:
                returned_action = self.players[i].play(
                    unit_id=self.unit_id[day][0],
                    unit_pos=self.unit_pos[day][0],
                    map_states=self.map_states[day][0],
                    current_scores=self.player_score[day][0],
                    total_scores=self.player_total_score[day])

            if self.check_action(returned_action, day, i):
                returned_action = [(float(dist), float(angle)) for dist, angle in returned_action]
                for j in range(len(returned_action)):
                    if self.check_move(returned_action[j]):
                        distance, angle = returned_action[j]
                        self.logger.debug(
                            "Received Distance: {:.3f}, Angle: {:.3f} from {}".format(distance, angle,
                                                                                      self.player_names[i]))
                        self.move_unit(distance, angle, day, i, j)
                    else:
                        self.logger.info(
                            "{} {} failed since provided invalid move {} (must contain tuples of finite value)".format(
                                self.player_names[i], self.unit_id[day][0][i][j], returned_action[j]))
                        self.empty_move_unit(day, i, j)
            else:
                self.logger.info(
                    "{} failed since provided invalid action {}".format(self.player_names[i], returned_action))
                self.empty_move(day, i)

        # State/score after units have moved
        score, map_state = self.fast_map.update_map_state(day, 1, self.unit_pos)
        self.player_score[day][1] = score
        self.map_states[day][1] = map_state

        # for i in range(constants.no_of_players):
        #     self.check_path_home(day, i)
        self.fast_map.check_path_home(day, self.unit_pos, self.unit_id)

        # State/score at end of day (killed isolated units)
        score, map_state = self.fast_map.update_map_state(day, 2, self.unit_pos)
        self.player_score[day][2] = score
        self.map_states[day][2] = map_state

        # Total score
        for i in range(constants.no_of_players):
            self.player_total_score[day][i] = self.player_total_score[day-1][i] + self.player_score[day][2][i]

        print("Day {} complete".format(day))

    def spawn_new(self, day, id_name):
        for i in range(constants.no_of_players):
            self.unit_id[day][0][i].append(id_name)
            self.unit_pos[day][0][i].append(self.base[i])

    def check_action(self, returned_action, day, idx):
        if not returned_action:
            return False
        if not isinstance(returned_action[0], tuple):
            return False  # Ensure no one is using sympy

        is_valid = False
        if len(returned_action) == len(self.unit_id[day][0][idx]):
            is_valid = True

        return is_valid

    def check_move(self, move):
        if not move:
            return False
        is_valid = False
        if isiterable(move) and count_iterable(move) == 2:
            if all(x is not None and not np.isnan(x) and np.isfinite(x) for x in move):
                is_valid = True

        return is_valid

    def move_unit(self, distance, angle, day, idx, unit_idx):
        angle = float(angle)
        a = self.unit_pos[day][0][idx][unit_idx].x
        b = self.unit_pos[day][0][idx][unit_idx].y
        if distance > 1.0:
            distance = 1.0
            self.logger.debug("Distance rectified to max distance of 1 km")

        new_a = a + (distance * np.cos(angle))
        new_b = b + (distance * np.sin(angle))

        if new_a < 0:
            new_a = 0
            new_b = b + (-a * np.tan(angle))
        elif new_a >= 100:
            new_a = 99.99999999
            new_b = b + ((new_a-a) * np.tan(angle))

        if new_b < 0:
            new_b = 0
            new_a = a + (-b / np.tan(angle))
        elif new_b >= 100:
            new_b = 99.99999999
            new_a = a + ((new_b-b) / np.tan(angle))

        new_pos = Point(new_a, new_b)
        self.unit_id[day][1][idx].append(self.unit_id[day][0][idx][unit_idx])
        self.unit_pos[day][1][idx].append(new_pos)

    def empty_move(self, day, idx):
        self.unit_id[day][1][idx] = self.unit_id[day][0][idx][:]
        self.unit_pos[day][1][idx] = self.unit_pos[day][0][idx][:]

    def empty_move_unit(self, day, idx, unit_idx):
        self.unit_id[day][1][idx].append(self.unit_id[day][0][idx][unit_idx])
        self.unit_pos[day][1][idx].append(self.unit_pos[day][0][idx][unit_idx])

    def get_state(self, day, state=0):
        return_dict = dict()
        return_dict["day"] = day+1
        return_dict["day_states"] = constants.day_state_labels[state]
        return_dict["map_states"] = self.map_states[day][state]
        return_dict["player_names"] = self.player_names
        return_dict["player_score"] = self.player_score[day][state]
        return_dict["player_total_score"] = self.player_total_score[day]
        return_dict["unit_id"] = self.unit_id[day][state]
        return_dict["unit_pos"] = self.unit_pos[day][state]
        return return_dict
    
    def set_app(self, voronoi_app):
        self.voronoi_app = voronoi_app


class FastMapState:
    def __init__(self, map_size, base_loc):
        self.map_size = map_size
        self.spawn_loc = base_loc

        self.cell_origins = self._compute_cell_coords(map_size)
        self.occupancy_map = None  # 2d state map
        self._num_contested_pts_check = 100  # In case of dispute, how many cells at identical dist to check

    def update_map_state(self, day, state, unit_pos) -> tuple[list[int], list[list[int]]]:
        """Replaces func with same name in old logic
        Compute the occupancy map and scores
        """
        count = [0, 0, 0, 0]

        self.compute_occupancy_map(day, unit_pos, state)
        for player in range(4):
            count[player] = np.count_nonzero(self.occupancy_map == player)

        # Convert occupancy map to list map_states. -1 is disputed, 1-4 is players.
        map_state = (self.occupancy_map + 1)
        map_state = map_state.astype(int)  # occ map is uint8, so cannot represent neg int
        map_state[map_state == 5] = -1
        map_state_ = map_state.T.tolist()
        return count, map_state_

    def check_path_home(self, day, unit_pos, unit_id):
        """Replaces func with same name in old logic
        Updates unit list with valid units after killing isolated units
        """
        # Always check against units after moving
        units_alive, id_units_alive = self.remove_killed_units(day, 1, unit_pos, unit_id)

        # Hack - Can do this as list is a mutable object. But ideally, we'd return this and modify in the main class.
        # We're doing this to maintain structure of old code.
        unit_pos[day][2] = units_alive
        unit_id[day][2] = id_units_alive
        return

    def compute_occupancy_map(self, day, unit_pos, state, mask_grid_pos: np.ndarray = None):
        """Calculates the occupancy status of each cell in the grid

        Args:
            day: int. Which day to compute for.
            unit_pos: List - unit_pos[day][state][player][id] - shapely.geometry.Point
            state: int. Which state of day to use (init, after unit move, after unit kill).
                Note: state end is same as init state of next day.
            mask_grid_pos: Shape: [N, N]. If provided, only occupancy of these cells will be computed.
                Used when updating occupancy map.
        """
        # Which cells contain units
        occ_map = self.get_unit_occupied_cells(day, unit_pos, state)
        occ_cell_pts = self.cell_origins[occ_map < 4]  # list of unit
        player_ids = occ_map[occ_map < 4]  # Shape: [N,]. player id for each occ cell
        if player_ids.shape[0] < 1:
            raise ValueError(f"No units on the map")

        # Create KD-tree with all occupied cells
        kdtree = scipy.spatial.KDTree(occ_cell_pts)

        # Query points: coords of each cell whose occupancy is not computed yet
        if mask_grid_pos is None:
            mask = (occ_map > 4)  # Not computed points
        else:
            mask = (occ_map > 4) & mask_grid_pos
            occ_map = self.occupancy_map  # Update existing map
        candidate_cell_pts = self.cell_origins[mask]  # Shape: [N, 2]

        # For each query pt, get associated player (nearest cell with unit)
        # Find nearest 2 points to identify if multiple cells at same dist
        near_dist, near_idx = kdtree.query(candidate_cell_pts, k=2)

        # Resolve disputes for cells with more than 1 occupied cells at same distance
        disputed = np.isclose(near_dist[:, 1] - near_dist[:, 0], 0)
        disputed_cell_pts = candidate_cell_pts[disputed]  # Shape: [N, 2].
        if disputed_cell_pts.shape[0] > 0:
            # Distance of the nearest cell will be radius of our search
            radius_of_dispute = near_dist[disputed, 0]  # Shape: [N, ]
            occ_map = self._filter_disputes(occ_map, kdtree, disputed_cell_pts, radius_of_dispute, player_ids)

        # For the rest of the cells (undisputed), mark occupancy
        not_disputed_ids = player_ids[near_idx[~disputed, 0]]  # Get player id of the nearest cell
        not_disputed_cells = candidate_cell_pts[~disputed].astype(int)  # cell idx from coords of occupied cells
        occ_map[not_disputed_cells[:, 0], not_disputed_cells[:, 1]] = not_disputed_ids

        self.occupancy_map = occ_map
        return

    def _filter_disputes(self, occ_map, kdtree, disputed_cell_pts, radius_of_dispute, player_ids):
        """For each cell with multiple nearby neighbors, resolve dispute
        Split into a func for profiling
        """
        # Find all neighbors within a radius: If all neigh are same player, cell not disputed
        for disp_cell, radius in zip(disputed_cell_pts, radius_of_dispute):
            # Radius needs padding to conform to < equality.
            rad_pad = 1e-5
            d_near_dist, d_near_idx = kdtree.query(disp_cell,
                                                   k=self._num_contested_pts_check,
                                                   distance_upper_bound=radius + rad_pad)
            # We will get exactly as many points as requested. Extra points will have inf dist
            # Need to filter those points that are within radius (dist < inf).
            valid_pts = np.isfinite(d_near_dist)
            d_near_dist = d_near_dist[valid_pts]
            d_near_idx = d_near_idx[valid_pts]

            # Additional check - remove those points that are even 1e-5 distance further
            valid_pts = d_near_dist == d_near_dist[0]
            d_near_idx = d_near_idx[valid_pts]

            disputed_ids = player_ids[d_near_idx]  # Get player ids of the contesting cells
            all_same_ids = np.all(disputed_ids == disputed_ids[0])
            if all_same_ids:
                # Mark cell belonging to this player
                player = disputed_ids[0]
            else:
                # Mark cell as contested
                player = 4
            disp_cell = disp_cell.astype(int)  # Shape: [2,]
            occ_map[disp_cell[0], disp_cell[1]] = player

        return occ_map

    @staticmethod
    def _compute_cell_coords(map_size) -> np.ndarray:
        """Calculates the origin for each cell.
        Each cell is 1km wide and origin is in the center.

        Return:
            coords: Shape: [100, 100, 2]. Coords for each cell in grid.
        """
        x = np.arange(0.5, map_size, 1.0)
        y = x
        xx, yy = np.meshgrid(x, y, indexing="ij")
        coords = np.stack((xx, yy), axis=-1)
        return coords

    def get_unit_occupied_cells(self, day, unit_pos, state) -> np.ndarray:
        """Colculate which cells contain units and are occupied/disputed

        Args:
            unit_pos: List - unit_pos[day][state][player][id] - shapely.geometry.Point
            state: int. Which state of day to use (init, after unit move, after unit kill).

        Returns:
            unit_occupancy_map: Shape: [N, N]. Maps cells to players/dispute, before nearest neighbor calculations.
                0-3: Player. 4: Disputed. 5: Not computed
        """
        occ_map = np.ones((self.map_size, self.map_size), dtype=np.uint8) * 5  # 0-3 = player, 5 = not computed

        pts_hash = {}
        for player in range(4):
            for pt in unit_pos[day][state][player]:
                x, y = pt.coords[:][0]
                pos_grid = (int(y), int(x))  # Quantize unit pos to cell idx
                if pos_grid not in pts_hash:
                    pts_hash[pos_grid] = player
                    occ_map[pos_grid] = player
                else:
                    player_existing = pts_hash[pos_grid]
                    if player_existing != player:  # Same cell, multiple players
                        occ_map[pos_grid] = 4

        return occ_map

    def get_connectivity_map(self) -> np.ndarray:
        """Map of all cells that have a path to their respective home base.
        Returns:
            np.ndarray: Connectivity map: Valid cells are marked with the player number,
                others are set to 4 (disputed). Shape: [N, N]
        """
        occ_map = self.occupancy_map
        connected = np.ones_like(occ_map) * 4  # Default = disputed/empty
        for player in range(4):
            start = self.spawn_loc[player]
            start = (int(start[0]), int(start[1]))  # Convert to cell index

            if occ_map[start[1], start[0]] != player:
                # Player's home base no longer belongs to player
                continue

            h, w = occ_map.shape
            mask = np.zeros((h + 2, w + 2), np.uint8)

            floodflags = 8  # Check all 8 directions
            floodflags |= cv2.FLOODFILL_MASK_ONLY  # Don't modify orig image
            floodflags |= (1 << 8)  # Fill mask with ones where true

            num, im, mask, rect = cv2.floodFill(occ_map, mask, start, player, 0, 0, floodflags)
            connected[mask[1:-1, 1:-1].astype(bool)] = player
        return connected

    def remove_killed_units(self, day, state, unit_pos, unit_id) -> tuple[list[list[Point]], list[list[int]]]:
        """Remove killed units and recompute the occupancy map
        Returns:
            units_alive: List of alive units for each player. u[player][pt]
            id_units_alive: List of alive ids for each player. u[player][id]
        """
        connectivity_map = self.get_connectivity_map()

        # build new list of valid units
        units_alive = []  # List of list: u[player][pt]
        id_units_alive = []
        for player in range(4):
            units_alive_ = []
            id_units_alive_ = []
            for pt, id in zip(unit_pos[day][state][player], unit_id[day][state][player]):
                pos = pt.coords[:][0]
                pos_grid = (int(pos[1]), int(pos[0]))
                if connectivity_map[pos_grid] == player:
                    units_alive_.append(pt)
                    id_units_alive_.append(id)

            units_alive.append(units_alive_)
            id_units_alive.append(id_units_alive_)

        return units_alive, id_units_alive
