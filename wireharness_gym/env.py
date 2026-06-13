"""
V0.5 Wire Harness — fully learnable, ONE target configuration per episode.

Rework of v0_4_1 with every deterministic control path removed:
  - no deterministic near-target fallback (was: deterministic_move_t inside
    DETERMINISTIC_RADIUS), the policy controls all movers 100 % of the time;
  - no constraint override (was: choose_constraint_action could replace the
    learned action) — collision avoidance is learned from observation + reward.

One env instance trains ONE configuration ("stage" k = MOVER_TARGETS[k], the
k-th configuration row: one [x, y] per mover in mover order). Episodes start
from a random predecessor configuration (any other stage or the initial XML
layout), so per-stage policies can be chained in any order at execution time. The episode terminates when ALL movers are inside the
goal radius SIMULTANEOUSLY — the same criterion the chained test script uses
to hand over between per-stage models.

Observation (4*N + N*(N-1) + 74*N = 410 for N=5):
    For each mover i: (x - x_t, y - y_t, dist_target_norm, angle_target_norm)
        (raw signed offsets are mover MINUS target, v0_1_1 convention)
    For each pair i<j: ((xj-xi)/L_ij, (yj-yi)/L_ij)
        L_ij = cable rest length for cabled pairs, field diagonal otherwise
    For each mover i: 5×5 local obstacle map (25) + 7×7 local cable map (49)
        (a mover's own body and its OWN cables are free space in its maps —
        the trailing cable is unavoidable and would only add constant noise)

Action (2*N,): [vx0,vy0, ..., vxN,vyN] in [-1, 1] — fully learned.

Reward:
    -0.01 step cost
    +10 · Σ_i (progress toward target)
    -APF pairwise mover repulsion (< 0.5 m, capped at 20)
    -w_obstacle_map · Σ_i sum(5×5 map)    (ludwig_sb3 grid penalty; skipped
    -w_cable_map    · Σ_i sum(7×7 map)     for movers inside the goal radius)
    +10 per mover on first touch of its target
    +25 when all movers are inside the goal radius simultaneously (terminates)
"""

import itertools
import math
import os
import sys
import time

import numpy as np
import mujoco as mj
import imageio.v2 as imageio
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import render_map_panel
from model.mover import Mover


class WireHarnessEnv(gym.Env):
    """
    Gymnasium env: N movers, one target configuration (stage), physical cables.
    Fully learned control; collision awareness via local grid maps in the
    observation and grid penalties in the reward (ported from environment_ludwig_sb3).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    X_MIN, X_MAX = 0.0, 6.72
    Y_MIN, Y_MAX = 0.0, 3.84

    GRID_W = 72
    GRID_H = 50

    APF_INFLUENCE = 0.5
    APF_ETA       = 0.5

    SETTLE_TIME = 0.5   # seconds of physics after teleport-reset so cables relax

    def __init__(
        self,
        xml_path: str,
        mover_starts: list,
        mover_body_names: list,
        mover_joint_names: list,
        targets: list,
        stage: int = 0,
        simend: int = 60,
        vel: float = 2.0,
        goal_radius: float = 0.15,
        cable_pair_lengths: dict = None,
        field_diag: float = 7.74,
        w_obstacle_map: float = 0.1,
        w_cable_map: float = 0.005,
        cable_connect: list = None,
        cable_start_mu: list = None,
        render_mode: str = None,
    ):
        super().__init__()

        self.xml_path    = xml_path
        self.simend      = simend
        self.vel         = vel
        self.goal_radius = goal_radius
        self.render_mode = render_mode
        self.num_agents  = len(mover_body_names)
        self._frame_time = 1.0 / 60.0

        # Configuration-major: targets[k] = configuration k, one [x, y] per
        # mover in mover order (MOVER_TARGETS layout from config).
        self.targets  = [[list(t) for t in konf] for konf in targets]
        self.n_stages = len(self.targets)
        for k, konf in enumerate(self.targets):
            if len(konf) != self.num_agents:
                raise ValueError(
                    f"Configuration {k} has {len(konf)} targets, "
                    f"expected one per mover ({self.num_agents})."
                )
        if not 0 <= stage < self.n_stages:
            raise ValueError(f"stage must be in [0, {self.n_stages - 1}], got {stage}")
        self.stage = stage

        self.mover_starts   = [list(s) for s in mover_starts]
        self.w_obstacle_map = w_obstacle_map
        self.w_cable_map    = w_cable_map

        # ── MuJoCo ────────────────────────────────────────────────────────
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data  = mj.MjData(self.model)
        self.model.opt.timestep = 0.00025
        # Substep counts instead of absolute-time loops: MuJoCo 3.x auto-resets
        # (zeroing data.time) on NaN/huge state, which would turn a
        # `while data.time - simstart < dt` loop into a near-infinite
        # re-simulation and silently teleport everything to the rest pose.
        self._frame_substeps  = math.ceil(self._frame_time / self.model.opt.timestep)
        self._settle_substeps = math.ceil(self.SETTLE_TIME / self.model.opt.timestep)

        # ── Resolve body IDs ──────────────────────────────────────────────
        body_ids = []
        for name in mover_body_names:
            bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise RuntimeError(f"Body '{name}' not found in XML.")
            body_ids.append(bid)

        # Cable bodies for the global collision map (selected by name, not by
        # index range). The bodies adjacent to each platform attachment
        # (CABLE_START_MU from config_base / environment_ludwig_sb3) are
        # excluded so attachment stubs don't register as permanent collisions
        # right under the platforms; without that list, fall back to excluding
        # the welded B_first/B_last bodies only.
        excluded = set()
        for ids in (cable_start_mu or []):
            excluded.update(ids)
        self._cable_bodies = []
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if not name.startswith("Wire"):
                continue
            if excluded:
                if i in excluded:
                    continue
            elif name.endswith("B_first") or name.endswith("B_last"):
                continue
            self._cable_bodies.append((i, int(name[4])))

        # ── Instantiate movers ────────────────────────────────────────────
        cable_connect  = cable_connect  or [[] for _ in range(self.num_agents)]
        cable_start_mu = cable_start_mu or [[] for _ in range(self.num_agents)]
        self.movers = []
        for i in range(self.num_agents):
            m = Mover(
                env=self,
                mu_index=body_ids[i],
                mu_start=list(mover_starts[i]),
                mu_joint=mover_joint_names[i],
                mu_start_move=[0.0, 0.0],
                follow=False,
                max_dist=float("inf"),
                vel=vel,
                cable_connect=list(cable_connect[i]),
                cable_start_mu=list(cable_start_mu[i]),
            )
            m.set_target(*self.targets[self.stage][i])
            self.movers.append(m)

        # Convenience references (used in tests/callbacks) — all 5 movers
        self.mover1 = self.movers[0]
        self.mover2 = self.movers[1]
        self.mover3 = self.movers[2]
        self.mover4 = self.movers[3]
        self.mover5 = self.movers[4]

        # ── Global collision map (cables + movers) ────────────────────────
        self.collision_map = np.zeros((self.GRID_H, self.GRID_W), dtype=np.float32)

        # Pairwise offset normalisation: cable rest length where cabled,
        # field diagonal otherwise.
        cable_pair_lengths = cable_pair_lengths or {}
        self._pair_norm = {}
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                self._pair_norm[(i, j)] = float(
                    cable_pair_lengths.get((i, j),
                    cable_pair_lengths.get((j, i), field_diag))
                )

        # ── Episode state ─────────────────────────────────────────────────
        self._reached   = [False] * self.num_agents
        self.sim_step   = 0
        self._max_steps = int(simend * 60)

        # ── Spaces ────────────────────────────────────────────────────────
        self._n_obs = (4 * self.num_agents
                       + self.num_agents * (self.num_agents - 1)
                       + 74 * self.num_agents)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2 * self.num_agents,), dtype=np.float32
        )

        # ── Camera / renderer ─────────────────────────────────────────────
        self._video_w, self._video_h = 640, 352
        self.renderer = None
        if render_mode is not None:
            try:
                self.renderer = mj.Renderer(self.model,
                                            width=self._video_w,
                                            height=self._video_h)
            except Exception as e:
                print(f"[WireHarnessEnv] Renderer unavailable: {e}")
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.cam.azimuth   = 90.0
        self.cam.distance  = 4.5
        self.cam.elevation = -60.0
        self.cam.lookat    = np.array([3.36, 1.6, 0.0])

        self._video_writer = None
        self._video_path   = None

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mj.mj_resetData(self.model, self.data)
        self.sim_step = 0
        self._reached = [False] * self.num_agents

        starts = self._sample_starts(options)
        self._teleport_and_settle(starts)

        for i, m in enumerate(self.movers):
            m.reward_sum    = 0
            m.done          = False
            m.coords_x      = []
            m.coords_y      = []
            m.path          = []
            m.path_original = []
            m.set_target(*self.targets[self.stage][i])
            m.update_pos()

        self._update_collision_map()
        for m in self.movers:
            self._update_local_maps(m)

        return self._get_obs(), {}

    def set_stage(self, stage: int):
        """Switch the active target configuration mid-episode (chained execution)."""
        if not 0 <= stage < self.n_stages:
            raise ValueError(f"stage must be in [0, {self.n_stages - 1}], got {stage}")
        self.stage    = stage
        self._reached = [False] * self.num_agents
        for i, m in enumerate(self.movers):
            m.set_target(*self.targets[stage][i])

    def _sample_starts(self, options):
        """
        Start positions for this episode. Default: uniform over the initial
        XML layout and every configuration except this stage's own — covers
        every possible predecessor when stages are chained in arbitrary order.
        options={"start": "initial"} forces the XML layout (chained test).
        """
        if options and options.get("start") == "initial":
            return self.mover_starts
        candidates = [self.mover_starts] + [
            self.targets[j] for j in range(self.n_stages) if j != self.stage
        ]
        return candidates[self.np_random.integers(len(candidates))]

    def _teleport_and_settle(self, starts):
        """
        Place movers via their slide-joint qpos (XML rest pose equals
        mover_starts, so offset = desired − rest), then run SETTLE_TIME of
        physics with the platforms pinned so the cables relax to the new
        endpoints before the first observation.
        """
        for m, (sx, sy) in zip(self.movers, starts):
            self.data.joint(m.joint_x).qpos[0] = sx - m.mu_start[0]
            self.data.joint(m.joint_y).qpos[0] = sy - m.mu_start[1]
            self.data.joint(m.joint_x).qvel[0] = 0.0
            self.data.joint(m.joint_y).qvel[0] = 0.0
        mj.mj_forward(self.model, self.data)

        settle_start = self.data.time
        for _ in range(self._settle_substeps):
            mj.mj_step(self.model, self.data)
            # data.time going backwards = MuJoCo auto-reset fired (NaN/huge
            # state) and wiped the teleport — fall back to the XML rest layout.
            if (self.data.time < settle_start
                    or np.isnan(self.data.qpos).any()
                    or np.abs(self.data.qpos).max() > 1e6):
                mj.mj_resetData(self.model, self.data)
                mj.mj_forward(self.model, self.data)
                return
            for m in self.movers:   # pin platforms while the cables relax
                self.data.joint(m.joint_x).qvel[0] = 0.0
                self.data.joint(m.joint_y).qvel[0] = 0.0
        self.data.qvel[:] = 0.0

    def _mask_action(self, action: np.ndarray) -> np.ndarray:
        action = action.copy()
        for i, m in enumerate(self.movers):
            if m.x <= self.X_MIN and action[2*i]   < 0: action[2*i]   = 0.0
            if m.x >= self.X_MAX and action[2*i]   > 0: action[2*i]   = 0.0
            if m.y <= self.Y_MIN and action[2*i+1] < 0: action[2*i+1] = 0.0
            if m.y >= self.Y_MAX and action[2*i+1] > 0: action[2*i+1] = 0.0
        return action

    def step(self, action):
        action = np.atleast_1d(np.asarray(action, dtype=np.float32))
        if np.isnan(action).any():
            return self._get_obs(), -0.02, True, False, \
                {"sim_step": self.sim_step, "n_at_goal": 0, "nan_action": True}

        action = np.clip(action, -1.0, 1.0)
        action = self._mask_action(action)

        for m in self.movers:
            m.update_pos()
        prev_dists = [m.get_distance_target(norm=False) for m in self.movers]

        # ── Per-mover action: 100 % learned ───────────────────────────────
        for i, m in enumerate(self.movers):
            a = action[2*i: 2*i+2].tolist()
            length = math.sqrt(a[0]**2 + a[1]**2)
            if length > 1.0:
                a = [a[0] / length, a[1] / length]
            m.make_move(a)

        # ── Physics ───────────────────────────────────────────────────────
        # Fixed substep count + auto-reset detection: if MuJoCo's internal
        # checks fire (NaN/huge qpos/qvel/qacc) it calls mj_resetData itself,
        # which zeroes data.time and teleports everything to the rest pose —
        # detect that (time went backwards) and terminate as unstable instead
        # of silently continuing from a corrupted state.
        simstart = self.data.time
        unstable = False
        for _ in range(self._frame_substeps):
            mj.mj_step(self.model, self.data)
            if self.data.time < simstart:
                unstable = True
                break

        # Airbag for divergence below the engine's auto-reset thresholds
        _qpos = self.data.qpos
        _qvel = self.data.qvel
        _qacc = self.data.qacc
        if (unstable or
                np.isnan(_qpos).any() or np.isinf(_qpos).any() or np.abs(_qpos).max() > 1e6 or
                np.isnan(_qvel).any() or np.isinf(_qvel).any() or np.abs(_qvel).max() > 1e6):
            mj.mj_resetData(self.model, self.data)
            return (np.zeros(self._n_obs, dtype=np.float32), -20.0, True, False,
                    {"sim_step": self.sim_step, "n_at_goal": 0, "physics_unstable": True})
        if np.isnan(_qacc).any() or np.isinf(_qacc).any() or np.abs(_qacc).max() > 1e9:
            self.data.qvel[:] = 0.0

        self.sim_step += 1

        for m in self.movers:
            m.update_pos()
        self._update_collision_map()
        for m in self.movers:
            self._update_local_maps(m)

        curr_dists = [m.get_distance_target(norm=False) for m in self.movers]

        # ── Reward ────────────────────────────────────────────────────────
        reward = -0.01

        for prev, curr in zip(prev_dists, curr_dists):
            reward += (prev - curr) * 10.0

        for ma, mb in itertools.combinations(self.movers, 2):
            d_pair = math.dist([ma.x, ma.y], [mb.x, mb.y])
            if d_pair < self.APF_INFLUENCE:
                d_safe  = max(d_pair, 1e-3)
                penalty = 0.5 * self.APF_ETA * (1.0 / d_safe - 1.0 / self.APF_INFLUENCE) ** 2
                reward -= min(penalty, 20.0)

        # Grid penalties (ludwig_sb3): obstacles seen in the 5×5 map,
        # cables/movers seen in the 7×7 map. Movers already inside their goal
        # radius are exempt — the target formation itself dictates cable
        # proximity there (measured up to -0.86/step held at Konf 3), and
        # penalizing it would teach hovering OFF target instead of holding.
        for m, d in zip(self.movers, curr_dists):
            if d < self.goal_radius:
                continue
            reward -= self.w_obstacle_map * float(np.sum(m.mu_collision_map))
            reward -= self.w_cable_map    * float(np.sum(m.mu_cable_collision_map))

        # One-time per-mover bonus on first touch of the stage target
        for i, d in enumerate(curr_dists):
            if d < self.goal_radius and not self._reached[i]:
                reward += 10.0
                self._reached[i] = True

        # The configuration counts as reached only when ALL movers are inside
        # the goal radius at the same time — matches the chained-execution
        # hand-over criterion, so the policy must learn to hold the formation.
        terminated = all(d < self.goal_radius for d in curr_dists)
        truncated  = self.sim_step >= self._max_steps

        if terminated:
            reward += 25.0

        for m in self.movers:
            m.coords_x.append(m.x)
            m.coords_y.append(m.y)

        if self._video_writer is not None:
            self._capture_frame()

        info = {
            "sim_step":  self.sim_step,
            "stage":     self.stage,
            "n_at_goal": sum(d < self.goal_radius for d in curr_dists),
            **{f"dist_to_target_{i+1}": float(curr_dists[i])
               for i in range(self.num_agents)},
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None or self.renderer is None:
            return None
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        return self.renderer.render()

    def close(self):
        self.finish_video()

    # ──────────────────────────────────────────────────────────────────────
    # Collision maps (cables + movers, ported from environment_ludwig_sb3)
    # ──────────────────────────────────────────────────────────────────────

    # Table-edge bounds for the local maps: grid cells outside the table read
    # as wall. Table: x ∈ [0, 6.72] → grid x 0..67; y ∈ [0, 3.84] → grid y 0..38.
    _GRID_X_TABLE = 67
    _GRID_Y_TABLE = 38

    def _update_local_maps(self, m):
        """
        Env-owned port of Mover.lokal_collision_map() with two fixes:
        - the own-cable filter is ENABLED (a mover's trailing cables are
          unavoidable; mover.py has the filter commented out at line 275),
        - both maps use the same table-edge wall bounds (mover.py uses 38 for
          the 5×5 but 48 for the 7×7).
        5×5 map: solid obstacles near the mover (other movers, other cables, walls).
        7×7 map: wider cable-awareness window (other cables, movers, walls).
        Own body (mu_index) and own cables (cable_connect ids) are free space.
        """
        x_idx = int(round(m.x, 1) * 10)
        y_idx = int(round(m.y, 1) * 10)
        own = set(m.cable_connect)

        m.mu_collision_map = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                gy, gx = y_idx - 2 + i, x_idx - 2 + j
                if 0 <= gy <= self._GRID_Y_TABLE and 0 <= gx <= self._GRID_X_TABLE:
                    entry = self.collision_map[gy, gx]
                    if entry != 0 and entry != m.mu_index and entry not in own:
                        m.mu_collision_map[i, j] = 1
                else:
                    m.mu_collision_map[i, j] = 1

        m.mu_cable_collision_map = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                gy, gx = y_idx - 3 + i, x_idx - 3 + j
                if 0 <= gy <= self._GRID_Y_TABLE and 0 <= gx <= self._GRID_X_TABLE:
                    entry = self.collision_map[gy, gx]
                    if entry != 0 and entry != m.mu_index and entry not in own:
                        m.mu_cable_collision_map[i, j] = 1
                else:
                    m.mu_cable_collision_map[i, j] = 1

    def _update_collision_map(self):
        self.collision_map[:] = 0.0
        # Cables first: one cell per cable body, value = cable id (1..4).
        for bid, cable_id in self._cable_bodies:
            x, y, _ = self.data.xpos[bid]
            gx = int(round(x, 1) * 10)
            gy = int(round(y, 1) * 10)
            if 0 <= gy < self.GRID_H and 0 <= gx < self.GRID_W:
                self.collision_map[gy, gx] = cable_id
        # Movers second (3×3 safety margin, value = body id): a mover
        # overwrites its own cable's cells beneath it, so it does not see
        # itself as a permanent collision.
        for m in self.movers:
            x_idx = int(round(m.x, 1) * 10)
            y_idx = int(round(m.y, 1) * 10)
            for j in range(-1, 2):
                for k in range(-1, 2):
                    gy, gx = y_idx + j, x_idx + k
                    if 0 <= gy < self.GRID_H and 0 <= gx < self.GRID_W:
                        self.collision_map[gy, gx] = m.mu_index

    # ──────────────────────────────────────────────────────────────────────
    # Observation — fully learnable, no privileged/deterministic features
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        obs = []
        # Per-mover target features (v0_1_1 style: raw offsets + normalized)
        for m in self.movers:
            obs.append(m.get_distance_x(m.x_t))
            obs.append(m.get_distance_y(m.y_t))
            obs.append(m.get_distance_target(norm=True))
            obs.append(m.get_angle_target(norm=True))
        # Pairwise signed offsets, cable-length normalized (v0_3 style)
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                norm = self._pair_norm[(i, j)]
                obs.append((self.movers[j].x - self.movers[i].x) / norm)
                obs.append((self.movers[j].y - self.movers[i].y) / norm)
        # Per-mover local grid maps (ludwig_sb3): 5×5 obstacles + 7×7 cables
        for m in self.movers:
            obs.extend(m.mu_collision_map.flatten())
            obs.extend(m.mu_cable_collision_map.flatten())
        return np.array(obs, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Video
    # ──────────────────────────────────────────────────────────────────────

    def start_video(self, path: str, fps: int = 30):
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._video_path = path
        if self.renderer is None:
            try:
                self.renderer = mj.Renderer(self.model,
                                            width=self._video_w,
                                            height=self._video_h)
            except Exception as e:
                print(f"[Video] Renderer unavailable: {e}")
        try:
            self._video_writer = imageio.get_writer(
                path, fps=fps, codec="libx264", macro_block_size=1)
        except Exception as e:
            self._video_writer = None
            print(f"[Video] Could not open writer: {e}")

    def _capture_frame(self):
        if self._video_writer is None or self.renderer is None:
            return
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        left  = self.renderer.render()
        # render_map_panel expects mover-major targets ([per-mover list of
        # targets]); self.targets is configuration-major, so transpose.
        targets_by_mover = [
            [konf[i] for konf in self.targets] for i in range(self.num_agents)
        ]
        right = render_map_panel(
            mover_positions=[(m.x, m.y) for m in self.movers],
            targets_list=targets_by_mover,
            current_indices=[self.stage] * self.num_agents,
            width=self._video_w, height=self._video_h,
            show_only_current=True,
        )
        self._video_writer.append_data(np.concatenate([left, right], axis=1))

    def finish_video(self):
        if self._video_writer is not None:
            try:
                self._video_writer.close()
            finally:
                self._video_writer = None
            time.sleep(0.1)
            if self._video_path and os.path.exists(self._video_path):
                print(f"[Video] Saved: {self._video_path}")
