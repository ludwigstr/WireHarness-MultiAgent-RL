"""
WireHarnessEnv — Gymnasium wrapper for Multi-Agent Wire Harness Routing.

INSTALLATION (required — pip install git+... does NOT work):

    git clone https://github.com/ludwigstr/WireHarness-MultiAgent-RL.git
    cd WireHarness-MultiAgent-RL
    pip install -e .

Reason: environment.py, config.py, mover.py etc. are in the repo root and are
not bundled into the wheel. An editable install keeps _ROOT pointing at the
cloned directory where all source files are present.

Usage:
    import gymnasium as gym
    import wireharness_gym          # triggers registration

    env = gym.make("WireHarnessRouting-v0")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Or directly:
    from wireharness_gym import WireHarnessEnv
    env = WireHarnessEnv(target_config_idx=2)
"""

import os
import sys

import platform
if platform.system() != "Windows":
    os.environ.setdefault("MUJOCO_GL", "egl")

# With editable install (pip install -e .), _ROOT points to the cloned repo root
# where config.py, environment.py etc. live. pip install git+... breaks this.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

try:
    import config
    from environment import Environment
    from utils.calculations import calculate_center_vector, calculate_rotations, calc_center_dists
except ModuleNotFoundError as e:
    raise ImportError(
        "\n\nCould not import WireHarness source modules. "
        "This package requires an editable install from a local clone:\n\n"
        "    git clone https://github.com/ludwigstr/WireHarness-MultiAgent-RL.git\n"
        "    cd WireHarness-MultiAgent-RL\n"
        "    pip install -e .\n\n"
        f"(Original error: {e})"
    ) from None


class WireHarnessEnv(gym.Env):
    """
    Multi-Agent Wire Harness Routing in MuJoCo.

    Five AGV movers carry cable segments on a 3.5 × 3.5 m table and must
    navigate to one of five predefined target configurations without causing
    cable collisions. A shared policy controls all five movers simultaneously
    via a single 10-dimensional action.

    Observation space (31-dim, float32):
        [0:20]  Pairwise signed distances (x, y) between all C(5,2)=10 mover pairs
        [20:30] Distance-to-target and angle-to-target for each of the 5 movers
        [30]    Target configuration index (0–4), included as a float

    Action space (10-dim, float32 in [-1, 1]):
        Normalised (vx, vy) velocity command for each of the 5 movers.
        The environment clips actions to unit length before execution.

    Reward:
        -1  per step (time penalty)
        -0.1 × mover-collision-map sum  (overlapping movers)
        -0.005 × cable-collision-map sum (cable on mover)
        +25 when all movers reach their target (episode success)

    Args:
        target_config_idx: Target configuration to use (0–4).
                           Pass ``None`` to sample uniformly at random each episode.
        render_mode: ``"rgb_array"`` to enable frame rendering via ``render()``,
                     ``None`` (default) for headless operation.
        xml_path: Path to the MuJoCo XML model file.
                  Defaults to the bundled ``WireHarness012e.xml``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": config.VIDEO_FPS}

    def __init__(
        self,
        target_config_idx: Optional[int] = 0,
        render_mode: Optional[str] = None,
        xml_path: Optional[str] = None,
    ):
        super().__init__()

        if target_config_idx is not None:
            assert 0 <= target_config_idx <= len(config.MOVER_TARGETS) - 1, (
                f"target_config_idx must be 0–{len(config.MOVER_TARGETS)-1} or None, "
                f"got {target_config_idx}"
            )
        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f"render_mode must be one of {self.metadata['render_modes']}"
        )

        self.target_config_idx = target_config_idx
        self.render_mode = render_mode
        self._xml_path = xml_path or config.XML_PATH

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config.STATE_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(config.ACTION_SIZE,),
            dtype=np.float32,
        )

        self._env: Optional[Environment] = None
        self._cfg_idx: int = 0
        self._vc = [0.0, 0.0]
        self._clockwise = False
        self._angles_t = [0.0] * config.NUM_MOVERS
        self._init_dists = [1.0] * config.NUM_MOVERS
        self._move_start = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_env(self) -> Environment:
        init_t = config.MOVER_TARGETS[0]
        dummy_wp = [[1.0] * 4 for _ in range(config.NUM_MOVERS)]
        env = Environment(
            xml_path=self._xml_path,
            num_agents=config.NUM_MOVERS,
            mu_index=config.BODY_IDS,
            mu_start=config.MU_START,
            mu_joints=config.JOINT_NAMES,
            mu_target=init_t,
            mu_start_move=config.MU_START_MOVE,
            mu_follow=config.MU_FOLLOW,
            mu_max_dist=config.MAX_DIST,
            simend=config.SIMEND,
            table_size=config.TABLE_SIZE,
            vel=config.VEL,
            start_sequence=config.START_SEQUENCE,
            cable_start=config.CABLE_START,
            cable_connect=config.CABLE_CONNECT,
            neighbor=config.NEIGHBOR,
            mu_target2=init_t,
            mu_target3=init_t,
            mu_target4=init_t,
            mu_target5=init_t,
            waypoint=dummy_wp,
            cable_end=config.CABLE_END,
            cable_start_mu=config.CABLE_START_MU,
            online_visualizer=False,
        )
        env.sim_config_body()
        return env

    def _set_target(self) -> None:
        for i, tgt in enumerate(config.MOVER_TARGETS[self._cfg_idx]):
            self._env.movers[i].set_target(tgt[0], tgt[1])

    def _compute_episode_params(self) -> None:
        """Compute geometric routing parameters once per episode."""
        for m in self._env.movers:
            m.update_pos()
        mu_pos = [[m.x, m.y] for m in self._env.movers]
        mu_t = config.MOVER_TARGETS[self._cfg_idx]
        try:
            vc, cs, ct = calculate_center_vector(mu_pos, mu_t)
            angles = calculate_rotations(mu_pos, mu_t, vc, ct)
            init_dists = calc_center_dists(mu_pos, cs)
            clockwise = not all(x > 0 for x in angles)
        except Exception:
            vc, clockwise = [0.0, 0.0], False
            angles = [0.0] * config.NUM_MOVERS
            init_dists = [1.0] * config.NUM_MOVERS
        self._vc = vc
        self._clockwise = clockwise
        self._angles_t = angles
        self._init_dists = init_dists

    def _make_obs(self, raw_states) -> np.ndarray:
        obs = np.array(raw_states, dtype=np.float32)
        if len(obs) < config.STATE_SIZE:
            obs = np.append(obs, float(self._cfg_idx))
        return obs

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.target_config_idx is None:
            self._cfg_idx = int(self.np_random.integers(0, len(config.MOVER_TARGETS)))
        else:
            self._cfg_idx = self.target_config_idx

        if self._env is None:
            self._env = self._build_env()

        self._env.reset()
        self._set_target()
        self._compute_episode_params()
        self._move_start = self._env.sim_step

        self._env.get_states()
        obs = self._make_obs(self._env.states)
        info = {
            "target_config_idx": self._cfg_idx,
            "target_positions": config.MOVER_TARGETS[self._cfg_idx],
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        raw_next, reward, done, stop, *_ = self._env.step(
            vc=self._vc,
            clockwise=self._clockwise,
            angles_t=self._angles_t,
            action_list=action,
            init_dists=self._init_dists,
            move_start=self._move_start,
            online_visualizer=False,
        )

        obs = self._make_obs(raw_next)
        nan_detected = not np.all(np.isfinite(obs))
        terminated = bool(done) or nan_detected
        truncated = False  # use gym.wrappers.TimeLimit for episode length

        info = {
            "target_config_idx": self._cfg_idx,
            "mover_distances": [
                abs(self._env.movers[i].get_distance_target(False))
                for i in range(config.NUM_MOVERS)
            ],
            "all_reached": bool(done) and not nan_detected,
            "nan_detected": nan_detected,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array" or self._env is None:
            return None
        self._env.renderer.update_scene(
            self._env.data, camera=self._env.cam, scene_option=self._env.opt
        )
        return self._env.renderer.render()

    def close(self):
        if self._env is not None:
            if self._env.video_writer is not None:
                self._env.finish_video()
            self._env = None
