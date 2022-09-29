# Voronoi
Project 4: Polygolf
Citation and License
This project belongs to Department of Computer Science, Columbia University. It may be used for educational purposes under Creative Commons with proper attribution and citation for the author TAs Naman Goyal (First Author), Griffin Adams and the Instructor, Prof. Kenneth Ross.

Summary
Course: COMS 4444 Programming and Problem Solving (Fall 2021)
Problem Description: http://www.cs.columbia.edu/~kar/4444f21/node21.html
Course Website: http://www.cs.columbia.edu/~kar/4444f21
University: Columbia University
Instructor: Prof. Kenneth Ross
Project Language: Python

TA Designer for this project
Naman Goyal

Teaching Assistants for Course
Griffin Adams
Naman Goyal
All course projects
Project 1: https://github.com/griff4692/chemotaxis
Project 2: https://github.com/thenamangoyal/icecream
Project 3: https://github.com/griff4692/coms4444_flowers/
Project 4: https://github.com/thenamangoyal/polygolf

Installation
Requires python3.6 or higher

Install simulator packages only

pip install -r requirements.txt
bash conda_requirements.sh
Install map generation packages. Note the --user option to avoid conflicts with system packages

pip install -r requirements_map.txt --user
Usage
Simulator
python main.py
Map Generation
Generating map and saving to <map_path>.json file

python gen_map.py -f <map_path>.json
Optional Flags
Map Generation
You can also specify the optional parameters below to change width, height and file path.

usage: gen_map.py [-h] [--file FILE] [--width WIDTH] [--height HEIGHT]

optional arguments:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  Path to export generated map
  --width WIDTH         Width
  --height HEIGHT       Height
Simulator
You can also specify the optional parameters below to disable GUI, disable browser launching, change port and address of server.

usage: main.py [-h] [--map MAP] [--skill SKILL] [--automatic] [--seed SEED]
               [--port PORT] [--address ADDRESS] [--no_browser] [--no_gui]
               [--log_path LOG_PATH] [--disable_timeout] [--disable_logging]
               [--players PLAYERS [PLAYERS ...]]

optional arguments:
  -h, --help            show this help message and exit
  --map MAP, -m MAP     Path to map json file
  --skill SKILL         Skill to use, don't specify to randomly choose between
                        min and max
  --automatic           Start playing automatically in GUI mode
  --seed SEED, -s SEED  Seed used by random number generator, specify 0 to use
                        no seed and have different random behavior on each
                        launch
  --port PORT           Port to start, specify -1 to auto-assign
  --address ADDRESS, -a ADDRESS
                        Address
  --no_browser, -nb     Disable browser launching in GUI mode
  --no_gui, -ng         Disable GUI
  --log_path LOG_PATH   Directory path to dump log files, filepath if
                        disable_logging is false
  --disable_timeout, -time
                        Disable Timeout in non GUI mode
  --disable_logging     Disable Logging, log_path becomes path to file
  --players PLAYERS [PLAYERS ...], -p PLAYERS [PLAYERS ...]
                        List of players space separated
Debugging
The code generates a log/debug.log (detailed), log/results.log (minimal) and log\<player_name>.log (logs from player) on every execution, detailing all the turns and steps in the game.
