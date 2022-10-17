#!/usr/bin/env bash

python main.py --dump_state --no_gui --last 1000 --spawn 20 --player1 8 --player2 7 --player3 6 --player4 5
echo "Rendering frames..."
python render_game.py
echo "Creating video..."
convert -delay 5 -loop 0 $(ls -1 render/*.png | sort -V) -quality 95 game.mp4
