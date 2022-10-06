import logging
import math
import os
import pickle
import time
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
        self.voronoi_app = None
        self.use_gui = not args.no_gui
        self.do_logging = not args.disable_logging
        if not self.use_gui:
            self.use_timeout = not args.disable_timeout
        else:
            self.use_timeout = False

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

        self.players = []
        self.player_names = []

        self.map_states = [[[[0 for z in range(constants.max_map_dim)] for k in range(constants.max_map_dim)] for j in
                            range(constants.day_states)] for i in range(self.last_day)]
        self.cell_units = [[[[0 for z in range(constants.max_map_dim)] for k in range(constants.max_map_dim)] for j in
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

        self.play_game()
        self.end_time = time.time()

        if self.use_gui:
            config = dict()
            config["address"] = args.address
            config["start_browser"] = not args.no_browser
            config["update_interval"] = 0.5
            config["userdata"] = (self, self.logger)
            if args.port != -1:
                config["port"] = args.port
            start(VoronoiApp, **config)
        else:
            self.logger.debug("No GUI flag specified")

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
        print("\nTime Elapsed - {:.3f}s".format(self.end_time-self.start_time))

        if args.dump_state:
            with open("game.pkl", "wb+") as f:
                pickle.dump(
                    {
                        "map_states": self.map_states,
                        "cell_units": self.cell_units,
                        "player_score": self.player_score,
                        "player_total_score": self.player_total_score,
                        "unit_id": self.unit_id,
                        "unit_pos": self.unit_pos,
                        "home_path": self.home_path,
                    },
                    f,
                )

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
            print("Day {} complete".format(i+1))

    def play_day(self, day):
        if day != 0:
            for i in range(constants.no_of_players):
                for j in range(len(self.unit_id[day - 1][2][i])):
                    self.unit_id[day][0][i].append(self.unit_id[day - 1][2][i][j])
                    self.unit_pos[day][0][i].append(self.unit_pos[day - 1][2][i][j])

        if day % self.spawn_day == 0:
            self.spawn_new(day, str((day // self.spawn_day) + 1))
            self.update_occupied_cells(day, 0)
            self.player_score[day][0] = self.update_map_state(day, 0)
        else:
            for i in range(constants.max_map_dim):
                for j in range(constants.max_map_dim):
                    self.map_states[day][0][i][j] = self.map_states[day - 1][2][i][j]
                    self.cell_units[day][0][i][j] = self.cell_units[day - 1][2][i][j]

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
                returned_action = [(float(dist), float(angle)) for dist, angle in returned_action]

            if self.check_action(returned_action, day, i):
                for j in range(len(returned_action)):
                    if self.check_move(returned_action[j]):
                        distance, angle = returned_action[j]
                        self.logger.debug(
                            "Received Distance: {:.3f}, Angle: {:.3f} from {}".format(distance, angle,
                                                                                      self.player_names[i]))
                        self.move_unit(distance, angle, day, i, j)
                    else:
                        self.logger.info(
                            "{} {} failed since provided invalid move {}".format(self.player_names[i],
                                                                                 self.unit_id[day][0][i][j],
                                                                                 returned_action[j]))
                        self.empty_move_unit(day, i, j)
            else:
                self.logger.info(
                    "{} failed since provided invalid action {}".format(self.player_names[i], returned_action))
                self.empty_move(day, i)

        self.update_occupied_cells(day, 1)
        self.player_score[day][1] = self.update_map_state(day, 1)

        for i in range(constants.no_of_players):
            self.check_path_home(day, i)

        self.update_occupied_cells(day, 2)
        self.player_score[day][2] = self.update_map_state(day, 2)
        for i in range(constants.no_of_players):
            self.player_total_score[day][i] = self.player_total_score[day-1][i] + self.player_score[day][2][i]

    def spawn_new(self, day, id_name):
        for i in range(constants.no_of_players):
            self.unit_id[day][0][i].append(id_name)
            self.unit_pos[day][0][i].append(self.base[i])

    def update_occupied_cells(self, day, state):
        for i in range(constants.no_of_players):
            for j in range(len(self.unit_id[day][state][i])):
                a = math.floor(self.unit_pos[day][state][i][j].x)
                b = math.floor(self.unit_pos[day][state][i][j].y)
                if self.cell_units[day][state][a][b] == 0:
                    self.cell_units[day][state][a][b] = i + 1
                elif self.cell_units[day][state][a][b] != i + 1:
                    self.cell_units[day][state][a][b] = -1

    def update_map_state(self, day, state):
        count = [0, 0, 0, 0]
        occupied_cells = []
        for i in range(constants.no_of_players):
            for j in range(len(self.unit_id[day][state][i])):
                a = math.floor(self.unit_pos[day][state][i][j].x)
                b = math.floor(self.unit_pos[day][state][i][j].y)
                if self.cell_units[day][state][a][b] > 0:
                    occupied_cells.append((a, b))

        if len(occupied_cells) == 0:
            return count

        for i in range(constants.max_map_dim):
            for j in range(constants.max_map_dim):
                if self.cell_units[day][state][i][j] != 0:
                    self.map_states[day][state][i][j] = self.cell_units[day][state][i][j]
                else:
                    a, b = occupied_cells[0]
                    a = int(a)
                    b = int(b)
                    min_dist = ((i - a) * (i - a)) + ((j - b) * (j - b))
                    min_dist_cells = [(a, b)]
                    min_dist_state = self.cell_units[day][state][a][b]
                    for k in range(1, len(occupied_cells)):
                        a, b = occupied_cells[k]
                        a = int(a)
                        b = int(b)
                        dist = ((i - a) * (i - a)) + ((j - b) * (j - b))
                        if dist < min_dist:
                            min_dist = dist
                            min_dist_cells = [(a, b)]
                            min_dist_state = self.cell_units[day][state][a][b]
                        elif dist == min_dist:
                            min_dist_cells.append((a, b))
                            if min_dist_state != self.cell_units[day][state][a][b]:
                                min_dist_state = -1

                    self.map_states[day][state][i][j] = min_dist_state

                if self.map_states[day][state][i][j] != -1:
                    count[self.map_states[day][state][i][j] - 1] += 1

        return count

    def check_action(self, returned_action, day, idx):
        if not returned_action:
            return False
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

    def check_path_home(self, day, idx):
        a = math.floor(self.base[idx].x)
        b = math.floor(self.base[idx].y)
        if self.map_states[day][1][a][b] == -1:
            self.home_path[day][idx][a][b] = True
        elif self.map_states[day][1][a][b] == (idx+1):
            self.path_map(day, idx, a, b)
            for i in range(len(self.unit_id[day][1][idx])):
                a = math.floor(self.unit_pos[day][1][idx][i].x)
                b = math.floor(self.unit_pos[day][1][idx][i].y)
                if self.home_path[day][idx][a][b]:
                    self.unit_id[day][2][idx].append(self.unit_id[day][1][idx][i])
                    self.unit_pos[day][2][idx].append(self.unit_pos[day][1][idx][i])
                else:
                    self.logger.info("{} unit {} has been cutoff from homebase".format(self.player_names[idx],
                                                                                       self.unit_id[day][1][idx][i]))



    def path_map(self, day, idx, x, y):
        stack = [(x, y)]
        while len(stack):
            a, b = stack.pop()
            a = int(a)
            b = int(b)
            self.home_path[day][idx][a][b] = True

            if a == 0:
                x_range = range(2)
            elif a == 99:
                x_range = range(-1, 1)
            else:
                x_range = range(-1, 2)

            if b == 0:
                y_range = range(2)
            elif b == 99:
                y_range = range(-1, 1)
            else:
                y_range = range(-1, 2)

            for i in x_range:
                for j in y_range:
                    if not self.home_path[day][idx][a + i][b + j]:
                        if self.map_states[day][1][a + i][b + j] == -1:
                            self.home_path[day][idx][a][b] = True
                        elif self.map_states[day][1][a + i][b + j] == (idx + 1):
                            stack.append((a + i, b + j))

    def get_state(self, day, state=0):
        return_dict = dict()
        return_dict["day"] = day+1
        return_dict["day_states"] = constants.day_state_labels[state]
        return_dict["map_states"] = self.map_states[day][state]
        return_dict["cell_units"] = self.cell_units[day][state]
        return_dict["player_names"] = self.player_names
        return_dict["player_score"] = self.player_score[day][state]
        return_dict["player_total_score"] = self.player_total_score[day]
        return_dict["unit_id"] = self.unit_id[day][state]
        return_dict["unit_pos"] = self.unit_pos[day][state]
        return return_dict
    
    def set_app(self, voronoi_app):
        self.voronoi_app = voronoi_app
