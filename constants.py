min_map_dim = 0
max_map_dim = 100
base = [(0.5, 0.5), (0.5, 99.5), (99.5, 99.5), (99.5, 0.5)]

player_color = ["#D55E00", "#CC79A7", "#004D79", "#F0E442"]
tile_color = ["#CC8C5A", "#DEA8C6", "#0072B2", "#FFF89E"]
dispute_color = "#666666"

no_of_players = 4
possible_players = ["d"] + list(map(str, range(1, 10)))
day_states = 3
day_state_labels = ["Start of Day", "During the Day", "End of Day"]

vis_width = 775
vis_height = 775
vis_width_ratio = 0.8
vis_height_ratio = 0.8
vis_padding = 0.02

timeout = 60 * 10
