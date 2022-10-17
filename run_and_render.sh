#!/usr/bin/env bash

python main.py --dump_state --no_gui --last 250 --spawn 5 --player1 5 --player2 1 --player3 1 --player4 1
echo "Rendering frames..."
python render_game.py
echo "Creating video..."
convert -delay 5 -loop 0 $(ls -1 render/*.png | sort -V) -quality 95 game.mp4
