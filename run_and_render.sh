#!/usr/bin/env bash

python3 main.py -p1 2 -p2 8 -p3 3 -p4 4 --no_gui --dump_state --last 250
echo "Rendering frames..."
python3 render_game.py
echo "Creating video..."
convert -delay 5 -loop 0 $(ls -1 render/*.png | sort -V) -quality 95 game.mp4
