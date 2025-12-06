# Skydio X2 Multi-Agent Simulation Framework

This repository is part of my Master’s personal project, where I explore topics in drone simulation, multi-agent control, and autonomous navigation. I built this project to document what I learned, experiment with ideas I find interesting, and share my code with anyone who is also passionate about drones, robotics, and control systems. This repository provides a complete framework for **multi-drone simulation**, **formation control**, **pure pursuit tracking**, and **path planning** using **MuJoCo (MJCF)** and Python.  
The project includes:

- Multi-agent control modules (consensus, quadcopter control, pure pursuit)
- Scenario definitions for autonomous drone missions
- A path-planning module (A* implementation)
- MuJoCo rendering and physics simulation tools
- Utilities for map building and multi-agent graph generation
- A full report documenting methodology

<p align="center">
  <img src="outputs/Multi-UAVs.png" width="100%">
</p>

---

## Features

### Multi-Agent Control (controls/)
- `consensus_controller.py` – consensus-based formation control  
- `quadcopter_controller.py` – low-level control for Skydio X2  
- `pure_pursuit.py` – reference tracking via look-ahead policy  

### Scenario-Based Simulation (scenario/)
- Bearing-only tracking  
- Drone tracking  
- Center tracking for multi-agent formations  
- Ready-to-run mission scripts  

### MuJoCo (MJCF) Simulation (mjcf/, mjc_simulate/)
- Custom MJCF model of Skydio X2  
- Multi-agent scenes  
- Scripts for real-time MJCF simulation and rendering  

### Path Planning (path_planning/)
- Fully custom A* implementation  
- Grid environment generator  
- Path visualization tools  

### Plotting & Post-Processing (plots/)
- Simulation trajectory plotting  
- Control input visualization  
- Tracking error visualization  

### Utilities (utilities/)
- Multi-drone waypoint generator  
- Map builder  
- Multi-agent graph generator  
- Waypoint visualization tools  

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/vietkhanh-nguyen/quadcopter_skydio_x2.git
cd quadcopter_skydio_x2
```

### 2. Running the Simulation
You could choose the desired scenario in main, and then run the main file
```bash
python execution/main.py
```
Keyboard-controlled demo
```bash
python execution/main_keyboard_x2.py
```
## Documentation
A complete project report explaining the algorithms and system design:
```bash
docs/report_final.pdf
```

## Example Outputs
Simulation results are located in `outputs/`