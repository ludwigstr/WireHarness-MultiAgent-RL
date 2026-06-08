"""
wireharness_gym — Gymnasium environment for Multi-Agent Wire Harness Routing.

Registers the following environment IDs:

    WireHarnessRouting-v0      target_config_idx=0  (fixed target)
    WireHarnessRouting-v1      target_config_idx=None  (random target per episode)

Installation (editable install required — pip install git+... does NOT work):
    git clone https://github.com/ludwigstr/WireHarness-MultiAgent-RL.git
    cd WireHarness-MultiAgent-RL
    pip install -e .

Quick start:
    import gymnasium as gym
    import wireharness_gym

    # Fixed target (config 0):
    env = gym.make("WireHarnessRouting-v0")

    # Random target sampled each episode:
    env = gym.make("WireHarnessRouting-v1")

    # Specific target config:
    from wireharness_gym import WireHarnessEnv
    env = WireHarnessEnv(target_config_idx=3)
"""

from gymnasium.envs.registration import register
from wireharness_gym.env import WireHarnessEnv

__all__ = ["WireHarnessEnv"]

register(
    id="WireHarnessRouting-v0",
    entry_point="wireharness_gym.env:WireHarnessEnv",
    max_episode_steps=750,
    kwargs={"target_config_idx": 0},
)

register(
    id="WireHarnessRouting-v1",
    entry_point="wireharness_gym.env:WireHarnessEnv",
    max_episode_steps=750,
    kwargs={"target_config_idx": None},
)
