#!/usr/bin/env bash

python main.py --no_gui --dump_state --last 100 -p1 4 -p2 3 -p4 1 -p3 2 --spawn 1
echo "Rendering frames..."
python render_game.py
echo "Creating video..."
convert -delay 5 -loop 0 $(ls -1 render/*.png | sort -V) -quality 95 game.mp4