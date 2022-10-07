#!/usr/bin/env bash

python main.py --last 400 --spawn 10 -p1 6 -p2 6 -p3 5 -p4 4 --no_gui --dump_state
echo "Rendering frames..."
python render_game.py
echo "Creating video..."
convert -delay 5 -loop 0 $(ls -1 render/*.png | sort -V) -quality 95 game.mp4