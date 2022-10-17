#!/usr/bin/env bash

python3 main.py --dump_state --no_gui --last 100 -p1 d -p2 9 -p3 d  -p4 d
echo "Rendering frames..."
python render_game.py
echo "Creating video..."
convert -delay 5 -loop 0 $(ls -1 render/*.png | sort -V) -quality 95 game.mp4
