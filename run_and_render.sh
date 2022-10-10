#!/usr/bin/env bash

python main.py --last 50 --player1 4 --no_gui --dump_state
echo "Rendering frames..."
python render_game.py
echo "Creating video..."
convert -delay 5 -loop 0 $(ls -1 render/*.png | sort -V) -quality 95 game.mp4
