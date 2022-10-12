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

DAY_STATE = 2
for day in range(-1, len(game_data["map_states"])):
    plt.clf()
    plt.title(f"Day {day + 1} - (t = {game_data['last_day']}, n = {game_data['spawn_day']})")
    ax = plt.gca()

    cmap = colors.ListedColormap([dispute_color] + tile_color)
    bounds = [-1, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    X, Y = np.meshgrid(list(range(100)), list(range(100)))
    if day == -1:
        plt.pcolormesh(
            X + 0.5,
            Y + 0.5,
            np.transpose(game_data["map_states"][0][0]),
            cmap=cmap,
            norm=norm,
        )
    else:
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
        if day == -1:
            for point in game_data["unit_pos"][0][0][p]:
                plt.plot(
                    point.x,
                    point.y,
                    color=player_color[p],
                    marker="o",
                    markersize=4,
                    markeredgecolor="black",
                )
        else:
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

    if day == -1:
        cell_values = [game_data["player_score"][0][0], [0 for i in range(4)]]
    else:
        cell_values = [game_data["player_score"][day][DAY_STATE], game_data["player_total_score"][day]]


    plt.table(
        cellText=cell_values,
        rowLabels=['Score', 'Total Score'],
        colLabels=game_data["player_names"],
        colColours=tile_color
    )

    plt.savefig(f"render/{day+1}.png")
