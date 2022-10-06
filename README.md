# Project 2: Voronoi

## Citation and License
This project belongs to Department of Computer Science, Columbia University. It may be used for educational purposes under Creative Commons **with proper attribution and citation** for the author TAs **Rohit Gopalakrishnan (First Author), Joe Adams and the Instructor, Prof. Kenneth Ross**.

## Summary

Course: COMS 4444 Programming and Problem Solving (Fall 2022)  
Problem Description: http://www.cs.columbia.edu/~kar/4444f22/node19.html  
Course Website: http://www.cs.columbia.edu/~kar/4444f22  
University: Columbia University  
Instructor: Prof. Kenneth Ross  
Project Language: Python

### TA Designer for this project

Rohit Gopalakrishnan

### Teaching Assistants for Course
1. Joe Adams
2. Rohit Gopalakrishnan

### All course projects
Project 1: https://github.com/ja3537/polygolf22  
Project 2: https://github.com/rvg-18227/Voronoi

## Installation

Requires **python3.6** or higher

Install simulator packages only

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

To generate the time lapse of the simulation, edit the run_and_render.sh file to add the necessary flags and run the command (ImageMagick needed):

```bash
bash run_and_render.sh
```

## Debugging

The code generates a `log/debug.log` (detailed), `log/results.log` (minimal) and `log\<player_name>.log` (logs from player) on every execution, detailing all the turns and steps in the game.
