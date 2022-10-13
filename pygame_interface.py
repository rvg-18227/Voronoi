import atexit
import datetime
from typing import Optional, List, Tuple

import cv2
import imageio_ffmpeg
import numpy as np
import matplotlib as mpl
import pygame
import shapely.geometry

import constants
from voronoi_game import VoronoiGame


class VoronoiInterface:
    def __init__(self, player_list, args, game_window_height=800, fps=16):
        """Interface for the Voronoi Game.
        Uses pygame to launch an interactive window

        Args:
            player_list: List of Player names. Must have 4 players.
            game_window_height: Width of the window the game will launch in

        Ref:
            Pygame Design Pattern: https://www.patternsgameprog.com/discover-python-and-patterns-8-game-loop-pattern/
        """
        atexit.register(self.cleanup)  # Calls destructor

        self.game_state = VoronoiGame(player_list, args)
        self.player_list = player_list  # Used to reset game
        self.args = args  # Used to reset game

        self.spawn_freq = self.game_state.spawn_day
        self.map_size = constants.max_map_dim
        scale_px = game_window_height // self.map_size
        self.total_days = self.game_state.last_day
        self.curr_day = -1
        self.logger = self.game_state.logger
        self.print_results = True
        # self.game_state = VoronoiEngine(player_list, map_size=map_size, total_days=total_days,
        #                                 save_video=None, spawn_freq=spawn_freq, player_timeout=player_timeout,
        #                                 seed=seed)
        self.renderer = VoronoiRender(map_size=self.map_size, scale_px=scale_px, unit_px=int(scale_px / 2))

        pygame.init()
        caption = "COMS 4444: Voronoi"
        pygame.display.set_caption(caption)
        self.running = True

        # pygame creates Surface objects on which it draws graphics. Surfaces can be layered on top of each other.
        # Window contains the map and a section to the right for text
        text_w = int(game_window_height * 0.77)  # Width of text info box
        s_width = self.renderer.img_w + text_w
        s_height = self.renderer.img_h

        # Main surface (game window). X-right, Y-down (not numpy format)
        flags = pygame.SHOWN  # | pygame.OPENGL
        self.screen = pygame.display.set_mode((s_width, s_height), flags=flags)
        # Text sub-surface
        self.text_box_surf = self.screen.subsurface(pygame.Rect((self.renderer.img_w, 0),
                                                                (text_w, self.renderer.img_h)))
        self.font = pygame.font.SysFont(None, 32)  # To create text
        self.info_end = ""  # Add text info

        # Game Map sub-surface
        self.occ_surf = self.screen.subsurface(pygame.Rect((0, 0), (self.renderer.img_w, self.renderer.img_h)))

        self.clock = pygame.time.Clock()
        self.fps = fps

        self.writer = None
        if args.out_video:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            player_str = ""
            for p in player_list:
                player_str += p
            self.video_path = f"videos/{now}-game-p{player_str}.mp4"
            self.frame = np.empty((s_width, s_height, 3), dtype=np.uint8)
            # disable warning (due to frame size not being a multiple of 16)
            self.writer = imageio_ffmpeg.write_frames(self.video_path, (s_width, s_height), ffmpeg_log_level="error",
                                                      fps=fps, quality=9)
            self.writer.send(None)  # Seed the generator

        # Game data
        self.reset = False
        self.pause = False

    def process_input(self):
        """Handle user inputs: events such as mouse clicks and key presses"""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False  # Close window
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    break

                elif event.key == pygame.K_r:
                    # Reset map
                    self.reset = True
                    self.logger.info(f"Reset the map")

                elif event.key == pygame.K_p:
                    # Reset map
                    self.pause = ~self.pause
                    print(f"Game paused: {bool(self.pause)}")

    def update(self):
        """Update the state of the game"""
        if self.reset:
            self.game_state = VoronoiGame(self.player_list, self.args)
            self.curr_day = -1
            self.reset = False
            self.print_results = True
            return

        if self.curr_day < self.total_days - 1:
            if not self.pause:
                self.curr_day += 1
                self.game_state.play_day(self.curr_day)
                self.info_end = ""
        else:
            self.info_end = "Game ended. Press R to reset, Esc to Quit"
            if self.print_results:
                self.game_state.print_results()

                if self.writer is not None:
                    self.writer.close()
                    self.writer = None
                    print(f"Saved video to: {self.video_path}")

                self.print_results = False  # Print results only once
            # self.running = False

    def render(self):
        if self.curr_day < 0:
            return

        self.screen.fill((255, 255, 255))  # Blank screen

        # Draw Map
        # Convert from game version to render version
        occupancy_map = np.array(self.game_state.map_states[self.curr_day][2], dtype=int).T
        occupancy_map -= 1
        occupancy_map[occupancy_map == -2] = 4
        unit_pos = self.game_state.unit_pos[self.curr_day][2]
        occ_img = self.renderer.get_colored_occ_map(occupancy_map, unit_pos)
        pygame.pixelcopy.array_to_surface(self.occ_surf, np.swapaxes(occ_img, 0, 1))

        self.draw_text()

        # Update the game window to see latest changes
        pygame.display.update()

        if self.writer is not None and not self.pause:
            if self.curr_day < self.total_days:
                # Don't record past end of game
                pygame.pixelcopy.surface_to_array(self.frame, self.screen)
                frame = np.ascontiguousarray(np.swapaxes(self.frame, 0, 1))
                self.writer.send(frame)

    def draw_text(self):
        # Draw Info text on screen surface
        self.text_box_surf.fill((247, 233, 218))  # White background for text
        box_rect = self.text_box_surf.get_rect()
        color = (0, 0, 0)
        pad_v = 0.1 * box_rect.height
        pad_top = 0.25 * box_rect.height
        pad_left = 0.2 * box_rect.width

        # Day count
        info_text = f"Day: {self.curr_day} / {self.total_days - 1}"
        text_surf = self.font.render(info_text, True, color)
        text_rect = text_surf.get_rect(midtop=box_rect.midtop)  # Position surface at center of text box
        text_rect.top += pad_top
        self.text_box_surf.blit(text_surf, text_rect.topleft)  # Draw text on text box

        # Player Count + msg
        total_score = self.game_state.player_total_score[self.curr_day]
        info_text = f"Player 1 ({self.game_state.player_names[0]}): {total_score[0]:,}\n" \
                    f"Player 2 ({self.game_state.player_names[1]}): {total_score[1]:,}\n" \
                    f"Player 3 ({self.game_state.player_names[2]}): {total_score[2]:,}\n" \
                    f"Player 4 ({self.game_state.player_names[3]}): {total_score[3]:,}\n"
        info_text += self.info_end
        text_lines = info_text.split("\n")
        for idx, line in enumerate(text_lines):
            text_surf = self.font.render(line, True, color)
            # Position Text left-aligned and spaced out
            text_rect = text_surf.get_rect(left=box_rect.left+pad_left, top=box_rect.top)
            text_rect.top += pad_top + (pad_v * (idx + 1))
            self.text_box_surf.blit(text_surf, text_rect.topleft)  # Draw text on text box

    def cleanup(self):
        # video - release and destroy windows
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            print(f"Saved video to: {self.video_path}")
        pygame.quit()

    def run(self):
        print(f"\nStarting pygame.")
        print(f"Keybindings:\n"
              f"  Esc: Quit the game.\n"
              f"  P: Pause the game.\n"
              f"  R: Reset game\n")

        while self.running:
            self.process_input()
            self.update()
            self.render()
            self.clock.tick(self.fps)  # Limit updates to 60 FPS. We're much slower.

        self.cleanup()


class VoronoiRender:
    def __init__(self, map_size: int, scale_px: int = 10, unit_px: int = 10):
        """Class to render the game map and parse screen/game coords

        Args:
            map_size: Width of the map, in km. Each cell will be 1km wide.
            scale_px: Each cell will be these many pixels wide
            unit_px: Size of each unit in pixels
        """
        self.map_size = map_size  # Width of the map in km. Each cell is 1km

        # Visualization
        self.scale_px = scale_px  # How many pixels wide each cell will be
        self.unit_size_px = unit_px
        self.grid_line_thickness = 2
        self.img_h = self.map_size * self.scale_px
        self.img_w = self.map_size * self.scale_px

        # Colors from: https://sashamaps.net/docs/resources/20-colors/
        self.player_back_colors = ['#fabed4', '#ffd8b1', '#aaffc3', '#42d4f4']
        player_colors = ['#e6194B', '#f58231', '#3cb44b', '#4363d8']
        self.player_colors = list(map(self._hex_to_rgb, player_colors))

    @staticmethod
    def _hex_to_rgb(col: str = "#ffffff"):
        return tuple(int(col.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

    def metric_to_px(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert metric unit pos (x, y) to pixel location on img of grid"""
        x, y = pos
        if not 0 <= x <= self.map_size:
            raise ValueError(f"x out of range [0, {self.map_size}]: {x}")
        if not 0 <= y <= self.map_size:
            raise ValueError(f"y out of range [0, {self.map_size}]: {y}")

        px, py = map(lambda z: int(round(z * self.scale_px)), [x, y])
        return px, py

    def px_to_metric(self, pos_px: Tuple) -> Tuple[float, float]:
        """Convert a pixel coord on map to metric
        Note: Pixels are in (row, col) format, transpose of XY Axes.
        """
        px, py = pos_px
        if not 0 <= px <= self.img_h:
            raise ValueError(f"x out of range [0, {self.img_h}]: {px}")
        if not 0 <= py <= self.img_w:
            raise ValueError(f"y out of range [0, {self.img_w}]: {py}")

        x, y = map(lambda z: round(z / self.scale_px, 2), [px, py])
        return x, y

    def get_colored_occ_map(self,
                            occ_map: np.ndarray,
                            # units: Optional[Dict[int, Dict]] = None,
                            units: Optional[List[List[shapely.geometry.Point]]] = None,
                            draw_major_lines: bool = True):
        """Visualizes an NxN Occupancy map for the voronoi game.

        Args:
            occ_map: Occupancy map. Shape: [n, n].
                Each cell is assigned a number from 0-4: 0-3 represents a player occupying it, 4 means contested
            units: If provided, will draw them on the map.
                Dict of units per player: u[player][pt]
            draw_major_lines: Draw grid lines

        Return:
            np.ndarray: The RGB image representing game state
        """
        assert len(occ_map.shape) == 2
        if occ_map.min() < 0 or occ_map.max() > 4:
            raise ValueError(f"Occupancy Map must contain values between 0-4")

        # Map cell states to colors
        # Colors from https://sashamaps.net/docs/resources/20-colors/
        cmap = mpl.colors.ListedColormap([*self.player_back_colors, '#ffffff'])
        norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], 5)  # 4 Discrete colors
        grid_rgb = cmap(norm(occ_map))[:, :, :3]
        grid_rgb = (grid_rgb * 255).astype(np.uint8)

        # Upsample img
        grid_rgb = cv2.resize(grid_rgb, None, fx=self.scale_px, fy=self.scale_px, interpolation=cv2.INTER_NEAREST)

        # Draw stuff
        if draw_major_lines:
            h, w, _ = grid_rgb.shape
            # Only show major grid lines (100x100 lines too fine) - max 10
            cols = min(10, occ_map.shape[1])
            col_line = (0, 0, 0)
            thickness = self.grid_line_thickness
            for x in np.linspace(start=int(thickness / 2), stop=w - int(thickness / 2), num=cols + 1):
                x = int(round(x))
                cv2.line(grid_rgb, (x, 0), (x, h), color=col_line, thickness=thickness)
                cv2.line(grid_rgb, (0, x), (w, x), color=col_line, thickness=thickness)

        if units is not None:
            for player_idx, pl_units in enumerate(units):
                for pt in pl_units:
                    # Draw Circle for each unit
                    pos = pt.coords[:][0]
                    pos_px = self.metric_to_px(pos)
                    cv2.circle(grid_rgb, pos_px, self.unit_size_px, self.player_colors[player_idx], -1)

        return grid_rgb
