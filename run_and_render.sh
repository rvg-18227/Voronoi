#!/usr/bin/env bash

python3 main.py --dump_state --no_gui --player1 8 --player2 d --player3 9  --last 50
echo "Rendering frames..."
python3 render_game.py
echo "Creating video..."
convert -delay 5 -loop 0 $(ls -1 render/*.png | sort -V) -quality 95 game.mp4
