"""
ENVIRONMENT - Main simulation environment for Multi-Agent Wire Harness Routing.
"""

import os
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import math
from mover import Mover
from utils.xml_utils import save_body_configuration, save_geometry_configuration
import imageio.v2 as imageio
import time

from utils.calculations import calculate_center_vector, calculate_rotations, calc_center_dists


class Environment:
    """
    MuJoCo-based simulation environment for Multi-Agent Wire Harness Routing.
    """

    def __init__(self, xml_path, num_agents, mu_index, mu_start, mu_joints, mu_target,
                 mu_start_move, mu_follow, mu_max_dist, simend, table_size, vel,
                 start_sequence, cable_start, cable_connect, neighbor, mu_target2,
                 mu_target3, mu_target4, mu_target5, waypoint, cable_end, cable_start_mu,
                 online_visualizer):
        """
        Initialises the simulation environment.
        """
        # ========== SIMULATION PARAMETERS ==========
        self.simend = simend
        self.table_size = table_size
        self.max_action = vel
        self.num_agents = num_agents
        self.start_sequence = start_sequence

        # ========== CABLE CONFIGURATION ==========
        self.cable_start = cable_start
        self.cable_connection = cable_connect
        self.cable_end = cable_end
        self.cable_start_mu = cable_start_mu

        # ========== MOVER CONFIGURATION ==========
        self.mu_index = mu_index
        self.mu_target = mu_target
        self.mu_start_move = mu_start_move
        self.neighbor = neighbor

        # ========== ADDITIONAL TARGETS ==========
        self.mu_target2 = mu_target2
        self.mu_target3 = mu_target3
        self.mu_target4 = mu_target4
        self.mu_target5 = mu_target5
        self.waypoint = waypoint

        # ========== COLLISION MAPS (50x72 grid) ==========
        self.collision_map = np.zeros((50, 72))
        self.cable_collision_map = np.zeros((50, 72))
        self.mover_collision_map = np.zeros((50, 72))
        self.path_map = np.zeros((50, 72))

        # ========== MUJOCO INITIALISATION ==========
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()

        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)

        # ========== VIDEO RECORDING SETUP ==========
        self.video_w, self.video_h = 640, 352
        self.video_writer = None
        self._video_path = None
        self._frame_count = 0

        # ========== CAMERA CONFIGURATION ==========
        self.cam.azimuth = 90.0
        self.cam.distance = 4.5
        self.cam.elevation = -60.0
        self.cam.lookat = np.array([3.36, 1.6, 0.0])

        # ========== ONLINE VISUALIZER ==========
        if online_visualizer:
            glfw.init()
            self.window = glfw.create_window(900, 400, "Live Simulation", None, None)
            glfw.make_context_current(self.window)
            self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
            self.scene = mj.MjvScene(self.model, maxgeom=10000)

        self.renderer = mj.Renderer(self.model, width=self.video_w, height=self.video_h)

        # ========== CREATE MOVER OBJECTS ==========
        self.movers = []
        self.dist_norm = []

        for i in range(num_agents):
            mover = Mover(
                self,
                mu_index[i],
                mu_start[i],
                mu_joints[i],
                mu_start_move[i],
                mu_follow[i],
                mu_max_dist[i],
                vel,
                cable_connect[i],
                cable_start_mu[i]
            )
            mover.set_target(mu_target[i][0], mu_target[i][1])
            self.movers.append(mover)

        # ========== INITIAL DISTANCES ==========
        for i in range(num_agents):
            dist = []
            for j in range(num_agents):
                dist.append(self.movers[i].get_distance(
                    self.movers[j].x,
                    self.movers[j].y,
                    0
                ))
            self.dist_norm.append(dist)

        for i in range(num_agents):
            self.movers[i].path = []

    def sim_config_geom(self):
        """Saves the current geometry configuration via xml_utils."""
        save_geometry_configuration(self, "data/simulation_config/geom_007.xml")

    def sim_config_body(self):
        """Saves the current body configuration via xml_utils."""
        save_body_configuration(self, "data/simulation_config/body_012b.xml")

    def calculate_collision_map(self):
        """Builds the 2D collision map of the workspace."""
        self.collision_map = np.zeros((50, 72))
        self.cable_collision_map = np.zeros((50, 72))
        self.mover_collision_map = np.zeros((50, 72))

        # Insert cables
        for i in range(self.model.nbody):
            if i > 0 and i < 91 and i not in self.cable_start:
                try:
                    x, y, _ = self.data.xpos[i]
                    x_idx = int(round(x, 1) * 10)
                    y_idx = int(round(y, 1) * 10)
                    cable_id = int(self.model.body(i).name[4])
                    self.collision_map[y_idx, x_idx] = cable_id
                    self.cable_collision_map[y_idx, x_idx] = cable_id
                except Exception:
                    pass

        # Insert movers
        for i in self.mu_index:
            x, y, _ = self.data.xpos[i]
            x_idx = int(round(x, 1) * 10)
            y_idx = int(round(y, 1) * 10)
            for j in range(-1, 2):
                for k in range(-1, 2):
                    try:
                        self.collision_map[y_idx + j, x_idx + k] = i
                        self.mover_collision_map[y_idx + j, x_idx + k] = i
                    except Exception:
                        pass

    def check_cable_on_mover(self, cable_connect):
        """Checks whether a cable runs over another mover."""
        self.calculate_collision_map()
        cable_collision_map_copy = np.copy(self.cable_collision_map)
        mover_collision_map_copy = np.copy(self.mover_collision_map)

        for cable in cable_connect:
            cable_collision_map_copy = np.copy(self.cable_collision_map)
            mover_collision_map_copy = np.copy(self.mover_collision_map)

            for y in range(cable_collision_map_copy.shape[0]):
                for x in range(cable_collision_map_copy.shape[1]):
                    if cable_collision_map_copy[y, x] != cable:
                        cable_collision_map_copy[y, x] = 0

            for i in range(self.num_agents):
                if cable in self.movers[i].cable_connect:
                    start = self.movers[i].mu_index
                    break

            for i in reversed(range(self.num_agents)):
                if cable in self.movers[i].cable_connect:
                    end = self.movers[i].mu_index
                    break

            for y in range(mover_collision_map_copy.shape[0]):
                for x in range(mover_collision_map_copy.shape[1]):
                    if mover_collision_map_copy[y, x] == start or mover_collision_map_copy[y, x] == end:
                        mover_collision_map_copy[y, x] = 0

            counter = 0
            for y in range(self.collision_map.shape[0]):
                for x in range(self.collision_map.shape[1]):
                    if mover_collision_map_copy[y, x] != 0 and cable_collision_map_copy[y, x] != 0:
                        counter += 1
                        if counter > 1:
                            return True
        return False

    def calculate_collision_maps(self):
        """Updates all collision maps."""
        self.calculate_collision_map()
        for i in range(self.num_agents):
            self.movers[i].update_pos()
            self.movers[i].lokal_collision_map()

    def get_distances(self):
        """Computes all pairwise distances and angles."""
        self.distances = []
        self.angles = []
        for i in range(self.num_agents):
            dist = []
            angle = []
            for j in range(self.num_agents):
                dist.append(self.movers[i].get_distance(
                    self.movers[j].x,
                    self.movers[j].y,
                    0
                ))
                angle.append(self.movers[i].get_angle(
                    self.movers[j].x,
                    self.movers[j].y,
                    False
                ))
            self.distances.append(dist)
            self.angles.append(angle)

    def get_states(self):
        """Assembles the state vector for the RL system."""
        for i in range(self.num_agents):
            self.movers[i].update_pos()

        self.states = []
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                self.states.append(self.movers[i].get_distance_x(self.movers[j].x))
                self.states.append(self.movers[i].get_distance_y(self.movers[j].y))
            self.states.append(self.movers[i].get_distance_target())
            self.states.append(self.movers[i].get_angle_target())

    def reset(self):
        """Resets the simulation to its initial state."""
        self.sim_step = 0
        self.target1 = False
        self.target2 = False
        self.target3 = False
        self.target4 = False
        self.target5 = False
        self.target6 = False
        self.deterministic = [False, False, False, False, False]

        _eq_name = 'eq_active0' if hasattr(self.model, 'eq_active0') else 'eq_active'
        _eq_arr = getattr(self.model, _eq_name, None)
        if _eq_arr is not None and len(_eq_arr) > 10:
            _eq_arr[10] = 0

        self.path_map = np.zeros((50, 72))

        for i in range(self.num_agents):
            self.movers[i].reward_sum = 0
            self.movers[i].done = False
            self.movers[i].coords = []
            self.movers[i].coords_x = []
            self.movers[i].coords_y = []
            self.movers[i].path = []
            self.movers[i].path_original = []
            self.movers[i].set_target(self.mu_target[i][0], self.mu_target[i][1])
            self.movers[i].wp_reached = False

        mj.mj_resetData(self.model, self.data)

        self.distances = []
        self.angles = []
        self.states = []
        self.actions = []
        self.done = False

        self.simstart = self.data.time

        for i in range(self.num_agents):
            self.movers[i].reward = 0
            self.movers[i].update_pos()

        self.get_states()
        return self.states

    def step(self, vc, clockwise, angles_t, action_list, init_dists, move_start, online_visualizer):
        """Main simulation step."""
        self.sim_step += 1
        self.distances = []
        self.angles = []
        self.states = []
        self.actions = []
        self.rewards = 0
        self.new_states = []
        def_action = [0, 0]
        stopped = [False, False, False, False, False]

        for i in range(self.num_agents):
            if self.check_cable_on_mover(self.movers[i].cable_connect):
                stopped[i] = True

        for i in range(self.num_agents):
            self.movers[i].done = False
            self.movers[i].reward = 0
            self.movers[i].update_pos()

        self.deterministic_action = [1, 1, 1, 1, 1]
        self.calculate_collision_maps()
        self.get_states()

        mu_pos = [
            [self.movers[0].x, self.movers[0].y],
            [self.movers[1].x, self.movers[1].y],
            [self.movers[2].x, self.movers[2].y],
            [self.movers[3].x, self.movers[3].y],
            [self.movers[4].x, self.movers[4].y],
        ]
        mu_t = [
            [self.movers[0].x_t, self.movers[0].y_t],
            [self.movers[1].x_t, self.movers[1].y_t],
            [self.movers[2].x_t, self.movers[2].y_t],
            [self.movers[3].x_t, self.movers[3].y_t],
            [self.movers[4].x_t, self.movers[4].y_t],
        ]

        _vc_on, cs, ct = calculate_center_vector(mu_pos, mu_t)
        angles = calculate_rotations(mu_pos, mu_t, vc, ct)
        calc_center_dists(mu_pos, cs)

        cs = np.asarray(cs, dtype=float)
        vc = np.asarray(vc, dtype=float)

        self.path_map = np.zeros((50, 72))
        for i in range(self.num_agents):
            self.movers[i].path = []

        for i in range(self.num_agents):
            if abs(self.movers[i].get_distance_target(False)) > 0.5:
                vm = np.asarray([self.movers[i].x, self.movers[i].y], dtype=float)
                pm = vm - cs
                if clockwise and angles[i] < 0:
                    pass
                elif clockwise and angles[i] > 0:
                    pass
                elif not clockwise and angles[i] > 0:
                    pass
                elif not clockwise and angles[i] < 0:
                    pass
                action = np.asarray([action_list[2 * i], action_list[2 * i + 1]], dtype=float)
            else:
                action = self.movers[i].deterministic_move_t()

            constraint_action = self.movers[i].choose_constraint_action(
                self.sim_step,
                self.movers[i].get_distance(self.movers[0].x, self.movers[0].y)
            )

            if not np.array_equal(constraint_action, [0, 0]) or stopped[i]:
                def_action = constraint_action
            else:
                def_action = action

            length = math.sqrt(def_action[0]**2 + def_action[1]**2)
            if length > 1:
                def_action[0] = def_action[0] / length
                def_action[1] = def_action[1] / length

            self.movers[i].make_move(def_action)
            self.actions.append(def_action)

        # ========== PHYSICS SIMULATION ==========
        self.simstart = self.data.time
        while self.data.time - self.simstart < 1.0 / 60.0:
            mj.mj_step(self.model, self.data)

        self.capture_frame()

        if online_visualizer:
            glfw.make_context_current(self.window)
            mj.mjr_setBuffer(0, self.context)
            width, height = glfw.get_framebuffer_size(self.window)
            viewport = mj.MjrRect(0, 0, width, height)
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        # ========== REWARD CALCULATION ==========
        self.get_distances()
        self.calculate_collision_maps()

        for i in range(self.num_agents):
            self.movers[i].reward -= 0.005 * np.sum(self.movers[i].mu_cable_collision_map)
            self.movers[i].reward -= 0.1 * np.sum(self.movers[i].mu_collision_map)

            if abs(self.movers[i].get_distance_target(False)) < 0.25:
                self.movers[i].done = True

            self.movers[i].reward_sum += self.movers[i].reward
            self.movers[i].reward_list.append(self.movers[i].reward_sum)
            self.rewards += self.movers[i].reward

        self.rewards -= 1

        # ========== NEW STATE ==========
        for i in range(self.num_agents):
            self.movers[i].update_pos()

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                self.new_states.append(self.movers[i].get_distance(
                    self.movers[j].x,
                    self.movers[j].y,
                    self.dist_norm[i][j]
                ))
                self.new_states.append(self.movers[i].get_angle(
                    self.movers[j].x,
                    self.movers[j].y
                ))
            self.new_states.append(self.movers[i].get_distance_target())
            self.new_states.append(self.movers[i].get_angle_target())

        # ========== STORE COORDINATES ==========
        for i in range(self.num_agents):
            self.movers[i].coords_x.append(self.data.xpos[self.movers[i].mu_index][0])
            self.movers[i].coords_y.append(self.data.xpos[self.movers[i].mu_index][1])

        stop = False

        # ========== DONE CHECK ==========
        self.done = all(self.movers[i].done for i in range(self.num_agents))
        if self.done:
            self.rewards += 25

        return self.new_states, self.rewards, self.done, stop

    # ========== VIDEO RECORDING ==========
    def start_video(self, path, fps=30):
        """Starts video recording."""
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._video_path = path
        self._frame_count = 0
        try:
            self.video_writer = imageio.get_writer(
                path, fps=fps, codec="libx264", macro_block_size=1
            )
        except Exception as e:
            self.video_writer = None
            print(f"[Video] Could not open MP4 writer: {e}")

    def capture_frame(self):
        """Captures a single video frame."""
        if self.video_writer is None:
            return
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        frame = self.renderer.render()
        self.video_writer.append_data(frame)
        self._frame_count += 1

    def finish_video(self):
        """Finalises video recording."""
        if self.video_writer is not None:
            try:
                self.video_writer.close()
            finally:
                self.video_writer = None
            time.sleep(0.2)
            ok = os.path.exists(self._video_path) and os.path.getsize(self._video_path) > 0
            if ok:
                print(f"[Video] Saved: {self._video_path}")
