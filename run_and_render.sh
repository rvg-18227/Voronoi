#!/usr/bin/env bash

python3 main.py -p1 8 -p2 8 -p3 8 -p4 8 --dump_state --no_gui --last 150 --spawn 1
echo "Rendering frames..."
python3 render_game.py
echo "Creating video..."
convert -delay 5 -loop 0 $(ls -1 render/*.png | sort -V) -quality 95 game.mp4
