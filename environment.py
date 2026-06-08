"""
ENVIRONMENT - Hauptsimulationsumgebung für Multi-Agent Wire Harness Routing
"""

import os
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import torch as T
import math
from mover import Mover
from utils.xml_utils import save_body_configuration, save_geometry_configuration
import imageio.v2 as imageio
import time

from utils.calculations import calculate_center_vector, signierter_winkel_2d, calculate_rotations, calculate_center_point, calculate_rotations_t, calc_center_dists

class Environment:
    """
    MuJoCo-basierte Simulationsumgebung für Multi-Agent Wire Harness Routing.
    """
    
    def __init__(self, xml_path, num_agents, mu_index, mu_start, mu_joints, mu_target, 
                 mu_start_move, mu_follow, mu_max_dist, simend, table_size, vel, 
                 start_sequence, cable_start, cable_connect, neighbor, mu_target2, 
                 mu_target3, mu_target4, mu_target5, waypoint, cable_end, cable_start_mu,
                 online_visualizer):
        """
        Initialisiert die Simulationsumgebung.
        """
        # ========== SIMULATIONSPARAMETER ==========
        self.simend = simend
        self.table_size = table_size
        self.max_action = vel
        self.num_agents = num_agents
        self.start_sequence = start_sequence
        
        # ========== KABEL-KONFIGURATION ==========
        self.cable_start = cable_start
        self.cable_connection = cable_connect
        self.cable_end = cable_end
        self.cable_start_mu = cable_start_mu
        
        # ========== MOVER-KONFIGURATION ==========
        self.mu_index = mu_index
        self.mu_target = mu_target
        self.mu_start_move = mu_start_move
        self.neighbor = neighbor
        
        # ========== WEITERE ZIELE ==========
        self.mu_target2 = mu_target2
        self.mu_target3 = mu_target3
        self.mu_target4 = mu_target4
        self.mu_target5 = mu_target5
        self.waypoint = waypoint
        
        # ========== COLLISION MAPS (50×72 Grid) ==========
        self.collision_map = np.zeros((50, 72))
        self.cable_collision_map = np.zeros((50, 72))
        self.mover_collision_map = np.zeros((50, 72))
        self.path_map = np.zeros((50, 72))
        
        # ========== MUJOCO INITIALISIERUNG ==========
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        
        # ========== VIDEO RECORDING SETUP ==========
        self.video_w, self.video_h = 640, 352
        # Dieser Renderer ist für die Videoaufnahme (capture_frame) zuständig
        self.video_writer = None
        self._video_path = None
        self._frame_count = 0
        
        # ========== KAMERA-KONFIGURATION ==========
        self.cam.azimuth = 90.0
        self.cam.distance = 4.5
        self.cam.elevation = -60.0
        self.cam.lookat = np.array([3.36, 1.6, 0.0])

        # ========== ONLINE VISUALIZER (Zusätzlich, falls gewünscht) ==========
        if online_visualizer:
            glfw.init()
            self.window = glfw.create_window(900, 400, "Live Simulation", None, None)
            glfw.make_context_current(self.window)
        
            # WICHTIG: Kontext erst erstellen, wenn Fenster aktiv ist
            self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
            self.scene = mj.MjvScene(self.model, maxgeom=10000)

        # Der Renderer für Videoaufnahmen sollte danach kommen
        self.renderer = mj.Renderer(self.model, width=self.video_w, height=self.video_h)
        
        # ========== MOVER-OBJEKTE ERSTELLEN ==========
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
        
        # ========== INITIALE DISTANZEN BERECHNEN ==========
        for i in range(num_agents):
            dist = []
            for j in range(num_agents):
                dist.append(self.movers[i].get_distance(
                    self.movers[j].x, 
                    self.movers[j].y, 
                    0
                ))
            self.dist_norm.append(dist)

        # PFADE INITIAL ZURÜCKSETZEN
        for i in range(num_agents):
            self.movers[i].path = []
    
    def sim_config_geom(self):
        """Wrapper für save_geometry_configuration aus xml_utils"""
        save_geometry_configuration(self, "data/simulation_config/geom_007.xml")
        
    def sim_config_body(self):
        """Wrapper für save_body_configuration aus xml_utils"""
        save_body_configuration(self, "data/simulation_config/body_012b.xml")
    
    def calculate_collision_map(self):
        """
        Erstellt eine 2D-Kollisionskarte des Arbeitsbereichs.
        """
        # Maps zurücksetzen
        self.collision_map = np.zeros((50, 72))
        self.cable_collision_map = np.zeros((50, 72))
        self.mover_collision_map = np.zeros((50, 72))

        # ========== KABEL IN COLLISION MAP EINTRAGEN ==========
        for i in range(self.model.nbody):
            if i > 0 and i < 91 and i not in self.cable_start:
                try:
                    x, y, _ = self.data.xpos[i]
                    x_idx = int(round(x, 1) * 10)
                    y_idx = int(round(y, 1) * 10)

                    cable_id = int(self.model.body(i).name[4])
                    
                    self.collision_map[y_idx, x_idx] = cable_id
                    self.cable_collision_map[y_idx, x_idx] = cable_id
                except Exception as e:
                    pass

        # ========== MOVER IN COLLISION MAP EINTRAGEN ==========
        for i in self.mu_index:
            x, y, _ = self.data.xpos[i]
            x_idx = int(round(x, 1) * 10)
            y_idx = int(round(y, 1) * 10)

            for j in range(-1, 2):
                for k in range(-1, 2):
                    try:
                        self.collision_map[y_idx + j, x_idx + k] = i
                        self.mover_collision_map[y_idx + j, x_idx + k] = i
                    except Exception as e:
                        pass

    def calculate_collision_map_pathplanning(self):
        """
        Erstellt eine 2D-Kollisionskarte des Arbeitsbereichs.
        """
        # Maps zurücksetzen
        self.collision_map_p = np.zeros((50, 72))
        self.cable_collision_map_p = np.zeros((50, 72))
        self.mover_collision_map_p = np.zeros((50, 72))

        # ========== KABEL IN COLLISION MAP EINTRAGEN ==========
        for i in range(self.model.nbody):
            if i > 0 and i < 91 and i not in self.cable_start:
                try:
                    x, y, _ = self.data.xpos[i]
                    x_idx = int(round(x, 1) * 10)
                    y_idx = int(round(y, 1) * 10)

                    cable_id = int(self.model.body(i).name[4])
                    
                    self.collision_map_p[y_idx, x_idx] = cable_id
                    self.cable_collision_map_p[y_idx, x_idx] = cable_id

                    if i not in self.cable_end:
                        x_n, y_n, _ = self.data.xpos[i + 1]
                        x_n_idx = int(round(x_n, 1) * 10)
                        y_n_idx = int(round(y_n, 1) * 10)

                        x_delta = max(abs(x_idx - x_n_idx), 1)
                        y_delta = max(abs(y_idx - y_n_idx), 1)

                        if x_delta < y_delta:
                            for x in range(min(x_idx, x_n_idx), max(x_idx, x_n_idx) + 1):
                                for y in range(0, round(y_delta / x_delta)):
                                    self.collision_map_p[min(y_idx, y_n_idx) + y, x] = cable_id
                        else:
                            for y in range(min(y_idx, y_n_idx), max(y_idx, y_n_idx) + 1):
                                for x in range(0, round(x_delta / y_delta)):
                                    self.collision_map_p[y, min(x_idx, x_n_idx) + x] = cable_id

                except Exception as e:
                    pass

        # ========== MOVER IN COLLISION MAP EINTRAGEN ==========
        for i in self.mu_index:
            x, y, _ = self.data.xpos[i]
            x_idx = int(round(x, 1) * 10)
            y_idx = int(round(y, 1) * 10)

            for j in range(-1, 2):
                for k in range(-1, 2):
                    try:
                        self.collision_map_p[y_idx + j, x_idx + k] = i
                        self.mover_collision_map_p[y_idx + j, x_idx + k] = i
                    except Exception as e:
                        pass
            
        # ========== ERWEITERTE COLLISION-BEREICHE ==========
        for i in range(self.num_agents):
            for l in self.cable_connection[i]:
                found = False
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        try:
                            if self.collision_map_p[y_idx + j, x_idx + k] == l:
                                found = True
                        except Exception as e:
                                pass
                if found == False:
                    for j in range(-3, 4):
                        for k in range(-3, 4):
                            try:
                                if self.collision_map_p[y_idx + j, x_idx + k] == l:
                                    if j == -3:
                                        self.collision_map_p[y_idx - 2, x_idx + k] = l
                                    if j == 3:
                                        self.collision_map_p[y_idx + 2, x_idx + k] = l
                                    if k == -3:
                                        self.collision_map_p[y_idx + j, x_idx - 2] = l
                                    if k == 3:
                                        self.collision_map_p[y_idx + j, x_idx + 2] = l
                            except Exception as e:
                                pass
    
    def check_cable_on_mover(self, cable_connect):
        """
        Prüft ob ein Kabel über einen anderen Mover läuft.
        """
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
        """
        Wrapper-Funktion die alle Collision-Maps aktualisiert.
        """
        self.calculate_collision_map()
        
        for i in range(self.num_agents):
            self.movers[i].update_pos()
            self.movers[i].lokal_collision_map()
    
    def calculate_angle_list(self):
        """
        Berechnet Winkelbeziehungen zwischen benachbarten Movern.
        """
        for i in range(len(self.neighbor) - 1):
            dx1 = self.movers[self.neighbor[i]].x - self.movers[0].x
            dy1 = self.movers[self.neighbor[i]].y - self.movers[0].y
            dx2 = self.movers[self.neighbor[i + 1]].x - self.movers[0].x
            dy2 = self.movers[self.neighbor[i + 1]].y - self.movers[0].y
            
            phi1 = math.atan2(dy1, dx1)
            phi2 = math.atan2(dy2, dx2)
            delta_phi = phi1 - phi2
            
            if abs(delta_phi) < 0.1:
                fak = 1
                
                perp1 = [-dy1, dx1]
                perp2 = [-dy2, dx2]
                
                len1 = math.sqrt(perp1[0]**2 + perp1[1]**2)
                len2 = math.sqrt(perp2[0]**2 + perp2[1]**2)
                
                # SAFETY CHECK: Verhindere Division durch Null
                if len1 > 1e-6:  # Epsilon-Check
                    self.movers[self.neighbor[i]].action_phi1 = [
                        fak * perp1[0]/len1, 
                        fak * perp1[1]/len1
                    ]
                else:
                    self.movers[self.neighbor[i]].action_phi1 = [0, 0]
                
                if len2 > 1e-6:  # Epsilon-Check
                    self.movers[self.neighbor[i + 1]].action_phi2 = [
                        fak * -perp2[0]/len2, 
                        fak * -perp2[1]/len2
                    ]
                else:
                    self.movers[self.neighbor[i + 1]].action_phi2 = [0, 0]
            else:
                self.movers[self.neighbor[i]].action_phi1 = [0, 0]
                self.movers[self.neighbor[i + 1]].action_phi2 = [0, 0]
    
    def get_distances(self):
        """
        Berechnet alle paarweisen Abstände und Winkel.
        """
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
        """
        Erstellt den State-Vektor für das RL-System.
        """
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
        """
        Setzt die Simulation auf den Anfangszustand zurück.
        """
        self.sim_step = 0
        self.target1 = False
        self.target2 = False
        self.target3 = False
        self.target4 = False
        self.target5 = False
        self.target6 = False
        self.deterministic = [False, False, False, False, False]
        
        # MuJoCo 3.x: eq_active0; ältere Versionen: eq_active
        _eq_name = 'eq_active0' if hasattr(self.model, 'eq_active0') else 'eq_active'
        _eq_arr = getattr(self.model, _eq_name, None)
        if _eq_arr is not None and len(_eq_arr) > 10:
            _eq_arr[10] = 0

        # ========== PATH MAP ZURÜCKSETZEN ==========
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
        """
        HAUPTFUNKTION: Führt einen Simulationsschritt aus.
        
        NUR NOCH THETA* - SEQUENZIELLE BEWEGUNG
        Alle Agents planen JEDEN Step mit Theta*, Constraints können überschreiben.
        
        Args:
            mover: Bewegungssequenz (Liste von Mover-Indices)
            mode: Rotationsrichtung ('clockwise'/'counter-clockwise'/None)
            cx, cy: Zentrum für Kreisbewegung (0, 0 bei direkter Bewegung)
            move_start: Start-Zeitpunkt der Bewegung
            use_circular: True (Kreisbahn) / "sequential_direct" (direkt)
        """
        # ========== INITIALISIERUNG ==========
        self.sim_step += 1
        self.distances = []
        self.angles = []
        self.states = []
        self.actions = []
        self.rewards = 0
        self.new_states = []
        def_action = [0, 0]
        stopped = [False, False, False, False, False]

        # Kollisions-Check
        for i in range(self.num_agents):
            if self.check_cable_on_mover(self.movers[i].cable_connect):
                stopped[i] = True

        for i in range(self.num_agents):
            self.movers[i].done = False
            self.movers[i].reward = 0
            self.movers[i].update_pos()
        
        self.deterministic_action = [1, 1, 1, 1, 1]
        self.calculate_collision_maps()
        self.calculate_angle_list()
        self.get_states()

        mu_pos = [
                [self.movers[0].x, self.movers[0].y],
                [self.movers[1].x, self.movers[1].y],
                [self.movers[2].x, self.movers[2].y],
                [self.movers[3].x, self.movers[3].y],
                [self.movers[4].x, self.movers[4].y]
            ]
        
        mu_t = [
                [self.movers[0].x_t, self.movers[0].y_t],
                [self.movers[1].x_t, self.movers[1].y_t],
                [self.movers[2].x_t, self.movers[2].y_t],
                [self.movers[3].x_t, self.movers[3].y_t],
                [self.movers[4].x_t, self.movers[4].y_t]
            ]

        vc_on, cs, ct = calculate_center_vector(
            mu_pos,
            mu_t
        )

        angles = calculate_rotations(
            mu_pos,
            mu_t,
            vc,
            ct
        )

        dists = calc_center_dists(mu_pos, cs)

        cs = np.asarray(cs, dtype=float)    # cs - Center point

        vc = np.asarray(vc, dtype=float)    # vc - Translation vector

        mean_angle_t_step_mu = []
        mean_angle = np.mean(angles)

        for i in range(self.num_agents):
            mean_angle_t_step_mu.append(angles_t[i] / 150)
        
        # ========== PATH MAP ZURÜCKSETZEN (vor der Bewegungsplanung) ==========
        self.path_map = np.zeros((50, 72))

        # PFADE ALLER MOVER ZURÜCKSETZEN (werden gleich neu geplant)
        for i in range(self.num_agents):
            self.movers[i].path = []

        # if self.sim_step < 300:
        #     segment = round(self.sim_step / 30 - 0.5)
        #     action_code = action_list[segment]

        learn = False

        for i in range(self.num_agents):
            action = np.zeros(2)

            if abs(self.movers[i].get_distance_target(False)) > 0.5:
                learn = True
                vt = np.array(self.movers[i].deterministic_move_t())

                vm = np.asarray([self.movers[i].x, self.movers[i].y], dtype=float)
                # vt = np.asarray([self.movers[i].x_t, self.movers[i].y_t], dtype=float)

                pm = vm - cs        # pm - Vector to center
                if clockwise and angles[i] < 0:
                    v_rot = np.array([pm[1], -pm[0]])
                elif clockwise and angles[i] > 0:
                    v_rot = np.array([-pm[1], pm[0]])
                elif not clockwise and angles[i] > 0:
                    v_rot = np.array([-pm[1], pm[0]])
                elif not clockwise and angles[i] < 0:
                    v_rot = np.array([pm[1], -pm[0]])
            
                # if i == 0:
                #     # print("0 ", angles[0])
                #     # print("T_0", angles_t[0])
                #     # print("Mean ", np.mean(angles))
                #     # print("Factor1 ", abs(angles_t[i] - angles[i]) / angles_t[i])
                #     # print("Factor2 ", 1 / abs(np.mean(angles) - angles[i]))
                #     # action += abs(angles_t[i] - angles[i]) / angles_t[i] * vt / np.linalg.norm(vt) 
                #     action += 7 / abs(np.mean(angles) - angles[i]) * v_rot / np.linalg.norm(v_rot)
                # else:
                # action += abs(angles_t[i] - angles[i]) / angles_t[i] * vt / np.linalg.norm(vt) 
                # # action += 7 / abs(np.mean(angles) - angles[i]) * v_rot / np.linalg.norm(v_rot) 
                # action += 0.7 * action_list[2 * i] * v_rot / np.linalg.norm(v_rot) 
                # # action += 0.3 * action_list[i] * pm / np.linalg.norm(pm)
                # action += 0.3 * action_list[2 * i + 1] * pm / np.linalg.norm(pm)

                action = np.asarray([action_list[2 * i], action_list[2 * i + 1]], dtype=float)
            else:
                action = self.movers[i].deterministic_move_t()

            # ========== 2. CONSTRAINT-CHECK ==========
            constraint_action = self.movers[i].choose_constraint_action(
                self.sim_step, 
                self.movers[i].get_distance(self.movers[0].x, self.movers[0].y)
            )
            
            # ========== 3. ENTSCHEIDUNG: Constraint oder geplante Action? ==========
            if not np.array_equal(constraint_action, [0, 0]) or stopped[i]:
                # Constraint ist aktiv → überschreibe geplante Action
                def_action = constraint_action
            else:
                # Kein Constraint → verwende geplante/gelernte Action
                def_action = action
            
            # ========== 4. NORMALISIERUNG ==========
            length = math.sqrt(def_action[0]**2 + def_action[1]**2)
            if length > 1:
                def_action[0] = def_action[0] / length
                def_action[1] = def_action[1] / length
            
            # ========== 5. BEWEGUNG AUSFÜHREN ==========
            self.movers[i].make_move(def_action)
            self.actions.append(def_action)

        # ========== PHYSICS SIMULATION ==========
        self.simstart = self.data.time
        
        while (self.data.time - self.simstart < 1.0/60.0):
            mj.mj_step(self.model, self.data)

        # 1. Video-Frame aufnehmen (nutzt den Renderer-Kontext)
        self.capture_frame() 

        # 2. Online Visualisierung (Fenster)
        if online_visualizer:
            # ERZWINGE den Fokus auf das Fenster
            glfw.make_context_current(self.window)
            
            # Buffer explizit auf das Fenster binden
            mj.mjr_setBuffer(0, self.context) # 0 = mjFB_WINDOW
            
            width, height = glfw.get_framebuffer_size(self.window)
            viewport = mj.MjrRect(0, 0, width, height)

            # Szene komplett neu aus den Daten aufbauen
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            
            # Rendern
            mj.mjr_render(viewport, self.scene, self.context)
            
            # Puffer tauschen
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        
        # ========== REWARD CALCULATION ==========
        self.get_distances()
        self.calculate_collision_maps()

        col_maps = []

        for i in range(self.num_agents):
            # self.movers[i].reward -= self.movers[i].get_distance_target(False)

            self.movers[i].reward -= 0.005 * np.sum(self.movers[i].mu_cable_collision_map)
            self.movers[i].reward -= 0.1 * np.sum(self.movers[i].mu_collision_map)

            col_maps.append(self.movers[i].mu_collision_map)
            
            if abs(self.movers[i].get_distance_target(False)) < 0.25:
                self.movers[i].done = True
                # self.movers[i].reward += 3
            
            self.movers[i].reward_sum += self.movers[i].reward
            self.movers[i].reward_list.append(self.movers[i].reward_sum)
            
            self.rewards += self.movers[i].reward

        self.rewards -= 1 	# angles[0] / 10
        
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
        
        # ========== KOORDINATEN SPEICHERN ==========
        for i in range(self.num_agents):
            self.movers[i].coords_x.append(self.data.xpos[self.movers[i].mu_index][0])
            self.movers[i].coords_y.append(self.data.xpos[self.movers[i].mu_index][1])
        
        delta_x = [1, 1, 1, 1, 1]
        delta_y = [1, 1, 1, 1, 1]

        # ========== STUCK DETECTION ==========
        # for i in range(self.num_agents):
        #     if self.sim_step - move_start > 20:
        #         mean_x1 = sum(self.movers[i].coords_x[-20:-10]) / 10
        #         mean_x2 = sum(self.movers[i].coords_x[-10:]) / 10
        #         mean_y1 = sum(self.movers[i].coords_y[-20:-10]) / 10
        #         mean_y2 = sum(self.movers[i].coords_y[-10:]) / 10
        #         delta_x[i] = abs(mean_x1 - mean_x2)
        #         delta_y[i] = abs(mean_y1 - mean_y2)
                
        #         if (delta_x[i] < 0.03) and (delta_y[i] < 0.03):
        #             stopped[i] = True
        
        stop = False
        
        # ========== DONE CHECK ==========
        self.done = True
        for i in range(self.num_agents):
            if self.movers[i].done == False:
                self.done = False
        
        if self.done:
            # print("Done")
            self.rewards += 25
        
        for i in range(self.num_agents):
            if self.check_cable_on_mover(self.movers[i].cable_connect):
                stopped[i] = True
        
        # self.capture_frame()

        if self.sim_step > 20:
            dashboard1 = [self.sim_step,
                    round(self.sim_step / 30),
                    stopped,
                    stop,
                    "RED",
                    angles_t[0],
                    angles[0],
                    mean_angle_t_step_mu[0],
                    "----------------",
                    (angles_t[0] - self.sim_step * mean_angle_t_step_mu[0]),
                    (angles[0]),                    
                    ((angles_t[0] - self.sim_step * mean_angle_t_step_mu[0]) / (angles[0])),
                    "GREEN",
                    angles_t[1],
                    angles[1],
                    (angles_t[1] - angles[1]),
                    mean_angle_t_step_mu[1],
                    (angles_t[1] - self.sim_step * mean_angle_t_step_mu[1]),
                    "YELLOW",
                    angles_t[2],
                    angles[2],
                    (angles_t[2] - angles[2]),
                    mean_angle_t_step_mu[2],
                    (angles_t[2] - self.sim_step * mean_angle_t_step_mu[2]),
                    "PURPLE",
                    angles_t[3],
                    angles[3],
                    (angles_t[3] - angles[3]),
                    mean_angle_t_step_mu[3],
                    (angles_t[3] - self.sim_step * mean_angle_t_step_mu[3]),
                    "ORANGE",
                    angles_t[4],
                    angles[4],
                    (angles_t[4] - angles[4]),
                    mean_angle_t_step_mu[4],
                    (angles_t[4] - self.sim_step * mean_angle_t_step_mu[4])
                    ]
        else:
            dashboard1 = [self.sim_step]
        
        dashboard2 = [

                ]
        
        dashboard3 = [

                ]
        
        return self.new_states, self.rewards, self.done, stop, learn, dashboard1, dashboard2, dashboard3
    
    # ========== VIDEO RECORDING FUNKTIONEN ==========
    def start_video(self, path, fps=30):
        """Startet Video-Aufnahme"""
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
            print(f"[Video] Konnte MP4-Writer nicht öffnen: {e}")
    
    def capture_frame(self):
        """Nimmt einen Frame auf"""
        if self.video_writer is None:
            return
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        frame = self.renderer.render()
        self.video_writer.append_data(frame)
        self._frame_count += 1
    
    def finish_video(self):
        """Beendet Video-Aufnahme"""
        if self.video_writer is not None:
            try:
                self.video_writer.close()
            finally:
                self.video_writer = None
            time.sleep(0.2)
            ok = os.path.exists(self._video_path) and os.path.getsize(self._video_path) > 0
            if ok:
                print(f"[Video] Erfolgreich gespeichert: {self._video_path}")