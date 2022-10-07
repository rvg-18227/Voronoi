import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pickle
from os import makedirs, remove
from glob import glob

from constants import player_color, tile_color, dispute_color, base

with open("game.pkl", "rb") as f:
    game_data = pickle.load(f)

makedirs("render", exist_ok=True)

old_files = glob("render/*.png")
for f in old_files:
    remove(f)

DAY_STATE = 0
for day in range(len(game_data["map_states"])):
    plt.clf()
    plt.title(f"Day {day + 1}")
    ax = plt.gca()

    cmap = colors.ListedColormap([dispute_color] + tile_color)
    bounds = [-1, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    X, Y = np.meshgrid(list(range(100)), list(range(100)))
    plt.pcolormesh(
        X + 0.5,
        Y + 0.5,
        np.transpose(game_data["map_states"][day][DAY_STATE]),
        cmap=cmap,
        norm=norm,
    )

    # Bases
    for p in range(4):
        base_x, base_y = base[p]
        plt.plot(
            base_x,
            base_y,
            color=player_color[p],
            marker="s",
            markeredgecolor="black",
        )

    # Units
    for p in range(4):
        for point in game_data["unit_pos"][day][DAY_STATE][p]:
            plt.plot(
                point.x,
                point.y,
                color=player_color[p],
                marker="o",
                markersize=4,
                markeredgecolor="black",
            )

    plt.xticks(np.arange(0, 100, 10))
    plt.yticks(np.arange(0, 100, 10))
    plt.grid(color="black", alpha=0.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    ax.set_aspect(1)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.invert_yaxis()

    plt.savefig(f"render/{day}.png")
