from typing import Dict, Tuple, List, Optional

import cv2
import matplotlib as mpl
import numpy as np


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

    # noinspection PyArgumentList
    def get_colored_occ_map(self,
                            occ_map: np.ndarray,
                            units: Optional[List[List[List[Tuple]]]] = None,
                            draw_major_lines: bool = True):
        """Visualizes an NxN Occupancy map for the voronoi game.

        Args:
            occ_map: Occupancy map. Shape: [n, n].
                Each cell is assigned a number from 0-4: 0-3 represents a player occupying it, 4 means contested
            units: If provided, will draw them on the map.
                List of shapely Points representing unit pos: u[player][id][pos]].
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
            # self.unit_pos[day][0] =
            # units[player][pos]]
            for player in range(4):
                u_list = units[player]
                for pt in u_list:
                    pos = pt.coords[:][0]
                    # Draw Circle for each unit
                    print(f"pos: {pos}")
                    pos_px = self.metric_to_px(pos)
                    cv2.circle(grid_rgb, pos_px, self.unit_size_px, self.player_colors[player], -1)

        return grid_rgb
