
import os
import sys

# Ensure v0_MAPF root is importable when this package is loaded in a subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym


from config import (
    MOVER_STARTS as MOVER_STARTS_V0_5,
    MOVER_BODY_NAMES as MOVER_BODY_NAMES_V0_5,
    MOVER_JOINT_NAMES as MOVER_JOINT_NAMES_V0_5,
    MOVER_TARGETS as MOVER_TARGETS_V0_5,
    XML_PATH as XML_PATH_V0_5,
    SIMEND as SIMEND_V0_5,
    VEL as VEL_V0_5,
    GOAL_RADIUS as GOAL_RADIUS_V0_5,
    CABLE_PAIR_LENGTHS as CABLE_PAIR_LENGTHS_V0_5,
    FIELD_DIAG as FIELD_DIAG_V0_5,
    W_OBSTACLE_MAP as W_OBSTACLE_MAP_V0_5,
    W_CABLE_MAP as W_CABLE_MAP_V0_5,
    CABLE_CONNECT as CABLE_CONNECT_V0_5,
    CABLE_START_MU as CABLE_START_MU_V0_5,
)

if "WireHarness-v0" not in gym.envs.registry:
    gym.register(
        id="WireHarness-v0",
        entry_point="env.wireharness_gym:WireHarnessEnv",
        kwargs={
            "xml_path":           os.path.abspath(XML_PATH_V0_5),
            "mover_starts":       MOVER_STARTS_V0_5,
            "mover_body_names":   MOVER_BODY_NAMES_V0_5,
            "mover_joint_names":  MOVER_JOINT_NAMES_V0_5,
            "targets":            MOVER_TARGETS_V0_5,   # configuration-major: targets[k] = Konf k
            "stage":              0,   # override per training run: gym.make(..., stage=k)
            "simend":             SIMEND_V0_5,
            "vel":                VEL_V0_5,
            "goal_radius":        GOAL_RADIUS_V0_5,
            "cable_pair_lengths": CABLE_PAIR_LENGTHS_V0_5,
            "field_diag":         FIELD_DIAG_V0_5,
            "w_obstacle_map":     W_OBSTACLE_MAP_V0_5,
            "w_cable_map":        W_CABLE_MAP_V0_5,
            "cable_connect":      CABLE_CONNECT_V0_5,
            "cable_start_mu":     CABLE_START_MU_V0_5,
        },
        max_episode_steps=SIMEND_V0_5 * 60,   # 3600 steps per configuration
    )

