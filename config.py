"""
Configuration for the Wire Harness Routing MuJoCo environment.

MOVER ASSIGNMENT:
  Index 0: RED     – Platform P1 – Cables 1,2,3,4
  Index 1: GREEN   – Platform P2 – Cable 1
  Index 2: YELLOW  – Platform P3 – Cable 2
  Index 3: PURPLE  – Platform P6 – Cable 3
  Index 4: ORANGE  – Platform P7 – Cable 4
"""

import os

# =============================================================================
# STATE / ACTION SPACE
# =============================================================================

STATE_SIZE  = 31   # 5 movers × 6 features + 1 target index
ACTION_SIZE = 10   # normalised (vx, vy) per mover × 5

# =============================================================================
# SIMULATION
# =============================================================================

NUM_MOVERS     = 5      # number of movers
SIMEND         = 30     # max simulation time [s]
TABLE_SIZE     = 3.5    # workspace side length [m]
VEL            = 2      # mover base velocity [m/s]
START_SEQUENCE = 10     # time steps before movement starts
VIDEO_FPS      = 30     # frame rate for rendered videos

# =============================================================================
# MOVER GEOMETRY
# =============================================================================

# Start positions [x, y] in metres — order: RED, GREEN, YELLOW, PURPLE, ORANGE
MU_START = [
    [4.0, 3.0],
    [5.0, 2.0],
    [1.0, 2.3],
    [1.5, 1.8],
    [4.0, 0.5],
]

JOINT_NAMES = [
    "slide_joint1",   # RED
    "slide_joint2",   # GREEN
    "slide_joint3",   # YELLOW
    "slide_joint6",   # PURPLE
    "slide_joint7",   # ORANGE
]

BODY_IDS = [91, 99, 105, 110, 115]

# Initial movement direction [x, y] at episode start
MU_START_MOVE = [
    [0, 0],
    [-0.5, 0],
    [0.1, 0.5],
    [-0.1, 0.4],
    [-0.3, 0.3],
]

MU_FOLLOW = [False, True, True, True, True]   # True = follows mover 0
MAX_DIST  = [99, 1.3, 3.1, 2.7, 2.4]          # max distance to neighbours [m]; 99 = unlimited

# =============================================================================
# CABLE TOPOLOGY
# =============================================================================

CABLE_CONNECT = [
    [1, 2, 3, 4],
    [1],
    [2],
    [3],
    [4],
]

CABLE_START_MU = [
    [1, 2, 3, 38, 39, 40, 68, 69, 70, 71, 72, 73],
    [8, 9, 10],
    [11, 12, 13],
    [41, 42, 43],
    [88, 89, 90],
]

CABLE_START = [
    1, 2, 3, 8, 9, 10, 11, 12, 13,
    38, 39, 40, 41, 42, 43,
    68, 69, 70, 71, 72, 73,
    88, 89, 90,
]

CABLE_END = [10, 40, 70, 90]

NEIGHBOR = [1, 4, 3, 2, 1]   # neighbour index in the cable chain

# =============================================================================
# TARGET CONFIGURATIONS
# =============================================================================

# Five target layouts — format: [config_idx][mover_idx][x, y]
MOVER_TARGETS = [
    [[5.2, 2.8], [5.3, 1.6], [2.5, 2.8], [3.1, 1.5], [5.1, 0.5]],  # 0: top-right
    [[5.2, 0.6], [4.0, 0.3], [5.4, 3.4], [5.0, 3.0], [2.8, 1.1]],  # 1: bottom-right
    [[3.0, 2.8], [3.5, 1.6], [0.5, 2.8], [1.1, 1.5], [3.1, 0.4]],  # 2: centre
    [[3.2, 0.4], [2.0, 0.3], [3.4, 3.4], [3.0, 3.0], [0.8, 0.7]],  # 3: left
    [[1.2, 0.8], [1.0, 2.0], [4.2, 0.4], [3.6, 0.8], [1.4, 3.2]],  # 4: taping stations
]

# =============================================================================
# PATHS
# =============================================================================

# Resolved relative to this file — works regardless of working directory
PROJECT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(PROJECT_DIR, "data")
XML_PATH       = os.path.join(DATA_DIR, "WireHarness012e.xml")
SIM_CONFIG_DIR = os.path.join(DATA_DIR, "simulation_config")
