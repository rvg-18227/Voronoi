#!/usr/bin/env bash

python3 main.py -p1 2 -p2 9 --no_gui --dump_state --spawn 10 --last 100
echo "Rendering frames..."
python render_game.py
echo "Creating video..."
convert -delay 5 -loop 0 $(ls -1 render/*.png | sort -V) -quality 95 game.mp4