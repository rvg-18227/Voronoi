#!/usr/bin/env bash

python main.py --no_gui --dump_state --last 60 -p1 4 -p2 4 -p3 4 -p4 4 --spawn 3
echo "Rendering frames..."
python render_game.py
#echo "Creating video..."
#convert -delay 5 -loop 0 $(ls -1 render/*.png | sort -V) -quality 95 game.mp4