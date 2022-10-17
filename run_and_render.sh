#!/usr/bin/env bash

python3 main.py --dump_state --no_gui --last 200 --spawn 1 -p1 8 -p2 8 -p3 8  -p4 8
echo "Rendering frames..."
python3 render_game.py
echo "Creating video..."
convert -delay 5 -loop 0 $(ls -1 render/*.png | sort -V) -quality 95 game.mp4
