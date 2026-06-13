"""
VISUALIZATION UTILITIES - Funktionen aus dem Original-Skript
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio.v2 as imageio
import os
import numpy as np

# One color per mover (up to 5): RED, GREEN, YELLOW, PURPLE, ORANGE
_TARGET_COLORS = ["red", "green", "yellow", "purple", "orange"]

# Workspace bounds matching the MuJoCo XML
_X_MIN, _X_MAX = 0.0, 6.72
_Y_MIN, _Y_MAX = 0.0, 3.84


_MOVER_MARKERS = ["s", "D", "^", "P", "*"]  # square, diamond, triangle, plus, star


def render_map_panel(
    mover_positions_or_x=None,
    targets_list_or_y=None,
    current_indices_or_targets=None,
    current_target_idx=None,
    width: int = 640,
    height: int = 352,
    *,
    # new-style keyword args (used by v0.1+)
    mover_positions=None,
    targets_list=None,
    current_indices=None,
    show_only_current: bool = False,
) -> np.ndarray:
    """
    Render a top-down 2-D map panel showing mover positions and target markers.

    Supports two call styles:

    Old (v0, single mover):
        render_map_panel(x, y, targets, current_target_idx, width=..., height=...)

    New (v0.1+, N movers):
        render_map_panel(
            mover_positions=[(x1,y1),(x2,y2),...],
            targets_list=[[t1_1,...],[t2_1,...],...],
            current_indices=[idx1, idx2, ...],
            width=..., height=...,
            show_only_current=True,  # show only the next target per mover
        )
    """
    # Resolve call style
    if mover_positions is not None:
        pass  # new-style kwargs already set
    elif isinstance(mover_positions_or_x, (int, float)):
        # Old positional style: (x, y, targets, current_idx)
        mover_positions  = [(mover_positions_or_x, targets_list_or_y)]
        targets_list     = [current_indices_or_targets]
        current_indices  = [current_target_idx]
    else:
        # New positional style: (mover_positions, targets_list, current_indices)
        mover_positions  = mover_positions_or_x
        targets_list     = targets_list_or_y
        current_indices  = current_indices_or_targets

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax  = fig.add_axes([0.13, 0.14, 0.82, 0.68])

    legend_patches = []

    for mi, ((mx, my), tgts, cur_idx) in enumerate(
            zip(mover_positions, targets_list, current_indices)):
        marker      = _MOVER_MARKERS[mi % len(_MOVER_MARKERS)]
        mover_color = _TARGET_COLORS[mi % len(_TARGET_COLORS)]
        n           = len(tgts)

        if show_only_current:
            # Show only the current (next) target for this mover
            if cur_idx < n:
                tx, ty = tgts[cur_idx]
                ax.plot(tx, ty, marker=marker, color=mover_color,
                        markersize=11, alpha=1.0, zorder=3,
                        markeredgecolor="white", markeredgewidth=0.8)
                ax.text(tx, ty + 0.14, f"Config {cur_idx+1}", ha="center", va="bottom",
                        fontsize=6.5, color=mover_color, alpha=1.0, fontweight="bold")
        else:
            # Target markers — all use the mover's own color, faded when reached
            for j, (tx, ty) in enumerate(tgts):
                reached = j < cur_idx
                alpha   = 0.25 if reached else 1.0
                ax.plot(tx, ty, marker=marker, color=mover_color,
                        markersize=11, alpha=alpha, zorder=3,
                        markeredgecolor="white", markeredgewidth=0.8)
                ax.text(tx, ty + 0.14, f"Config {j+1}", ha="center", va="bottom",
                        fontsize=6.5, color=mover_color, alpha=alpha, fontweight="bold")

        # Mover position (white dot, edged in mover color)
        ax.plot(mx, my, "o", color="white", markersize=8, zorder=4,
                markeredgecolor=mover_color, markeredgewidth=1.5)

    # Axes styling
    ax.set_xlim(_X_MIN, _X_MAX)
    ax.set_ylim(_Y_MIN, _Y_MAX)
    ax.set_xlabel("X [m]", fontsize=8, color="white")
    ax.set_ylabel("Y [m]", fontsize=8, color="white")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    ax.set_facecolor("#1a1a2e")
    ax.grid(True, color="#333", linewidth=0.5, alpha=0.6)
    fig.patch.set_facecolor("#0f0f1a")

    # Title — show current target configuration
    parts = [f"Target Config {current_indices[i]+1}" for i in range(len(mover_positions))]
    ax.set_title(" | ".join(parts), fontsize=8, color="white", pad=4)

    fig.canvas.draw()
    canvas_w, canvas_h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = buf.reshape(canvas_h, canvas_w, 3)
    plt.close(fig)

    if img.shape[:2] != (height, width):
        from PIL import Image as _PIL
        img = np.array(_PIL.fromarray(img).resize((width, height), _PIL.LANCZOS))

    return img


def reward_plot(env, path):
    """
    ORIGINAL-FUNKTION aus dem Hauptskript.
    Erstellt ein Diagramm der Rewards über die Zeit für alle Mover.
    
    Was diese Funktion macht:
    1. Erstellt ein Matplotlib-Diagramm
    2. Plottet die Reward-Historie jedes Movers
    3. Verwendet die Mover-Farben (rot, grün, gelb, lila, orange)
    4. Speichert als PNG-Datei
    
    Args:
        env: Environment-Objekt mit den Movern und deren reward_list
        path: Ausgabepfad für die PNG-Datei
    """
    # X-Achse: Zeitschritte (gleich lang für alle Mover)
    x_values = range(len(env.movers[0].reward_list))

    # Erstelle Figure mit bestimmter Größe
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    
    # Plotte Reward-Verlauf für jeden Mover mit seiner Farbe
    ax1.plot(x_values, env.movers[0].reward_list, label='Reward Red', color='red')
    ax1.plot(x_values, env.movers[1].reward_list, label='Reward Green', color='green')
    ax1.plot(x_values, env.movers[2].reward_list, label='Reward Yellow', color='yellow')
    ax1.plot(x_values, env.movers[3].reward_list, label='Reward Purple', color='purple')
    ax1.plot(x_values, env.movers[4].reward_list, label='Reward Orange', color='orange')
    
    # Achsenbeschriftungen und Titel
    ax1.set_xlabel('Zeit / Episoden')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward-Verlauf')
    
    # Legende anzeigen
    ax1.legend()
    
    # Als Datei speichern
    fig1.savefig(path)
    
    # Figure schließen um Speicher freizugeben
    plt.close(fig1)


class VideoRecorder:
    """
    Klasse für Video-Aufnahme der Simulation.
    Basiert auf den Video-Funktionen aus dem Original Environment.
    """
    
    def __init__(self, width=640, height=352):
        """
        Initialisiert den Video-Recorder.
        
        Args:
            width: Video-Breite in Pixeln (Standard: 640)
            height: Video-Höhe in Pixeln (Standard: 352)
        """
        self.video_w = width
        self.video_h = height
        self.video_writer = None
        self._video_path = None
        self._frame_count = 0
    
    def start_video(self, path, fps=30):
        """
        ORIGINAL-FUNKTION aus Environment.
        Startet die Video-Aufnahme der Simulation.
        
        Was diese Funktion macht:
        1. Konvertiert Pfad zu absolutem Pfad
        2. Erstellt Verzeichnisse falls nötig
        3. Initialisiert imageio Writer mit H.264 Codec
        4. Setzt Frame-Counter zurück
        
        Args:
            path: Ausgabepfad für das Video (z.B. "videos/simulation.mp4")
            fps: Frames pro Sekunde (Standard: 30)
        """
        # Absoluten Pfad verwenden für Zuverlässigkeit
        path = os.path.abspath(path)
        
        # Verzeichnis erstellen falls nicht vorhanden
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Pfad und Frame-Counter speichern
        self._video_path = path
        self._frame_count = 0
        
        try:
            # Video-Writer mit H.264 Codec initialisieren
            # macro_block_size=1 für bessere Qualität bei kleinen Details
            self.video_writer = imageio.get_writer(
                path, 
                fps=fps, 
                codec="libx264",  # H.264 Codec für MP4
                macro_block_size=1
            )
            print(f"[Video] Recording started → {path} ({self.video_w}x{self.video_h} @ {fps}fps)")
        except Exception as e:
            # Falls Video-Writer nicht erstellt werden kann
            self.video_writer = None
            print(f"[Video] Konnte MP4-Writer nicht öffnen: {e}")
    
    def capture_frame(self, renderer, data, cam, opt):
        """
        ORIGINAL-FUNKTION aus Environment.
        Nimmt einen Frame für das Video auf.
        
        Was diese Funktion macht:
        1. Prüft ob Video-Writer aktiv ist
        2. Aktualisiert die MuJoCo-Szene
        3. Rendert den Frame
        4. Fügt Frame zum Video hinzu
        5. Erhöht Frame-Counter
        
        Args:
            renderer: MuJoCo Renderer-Objekt
            data: MuJoCo Simulationsdaten
            cam: MuJoCo Kamera-Objekt
            opt: MuJoCo Visualisierungsoptionen
        """
        # Nur aufnehmen wenn Writer aktiv
        if self.video_writer is None:
            return
        
        # MuJoCo-Szene aktualisieren mit aktuellen Daten
        renderer.update_scene(data, camera=cam, scene_option=opt)
        
        # Frame rendern (gibt numpy array zurück)
        frame = renderer.render()
        
        # Frame zum Video hinzufügen
        self.video_writer.append_data(frame)
        
        # Frame-Counter erhöhen für Statistik
        self._frame_count += 1
    
    def finish_video(self):
        """
        ORIGINAL-FUNKTION aus Environment.
        Beendet die Video-Aufnahme und speichert die Datei.
        
        Was diese Funktion macht:
        1. Schließt den Video-Writer (speichert die Datei)
        2. Wartet kurz damit Datei vollständig geschrieben wird
        3. Prüft ob Datei erfolgreich erstellt wurde
        4. Gibt Erfolgsmeldung oder Fehlermeldung aus
        """
        if self.video_writer is not None:
            try:
                # Video-Writer schließen - das speichert die Datei
                self.video_writer.close()
            finally:
                # Writer auf None setzen für sauberen Zustand
                self.video_writer = None
            
            # Kurz warten damit OS die Datei fertig schreibt
            import time
            time.sleep(0.2)
            
            # Prüfen ob Datei existiert und Größe > 0 hat
            ok = os.path.exists(self._video_path) and os.path.getsize(self._video_path) > 0
            
            # Status-Meldung ausgeben
            print(f"[Video] Recording finished ({self._frame_count} Frames) → {self._video_path} "
                  f"{'(OK)' if ok else '(FEHLER: Datei fehlt/leer)'}")


def integrate_video_recording(env):
    """
    Hilfsfunktion um Video-Recording in Environment zu integrieren.
    
    Was diese Funktion macht:
    - Fügt die Video-Methoden zum Environment hinzu
    - Initialisiert benötigte Variablen
    
    Args:
        env: Environment-Objekt
    """
    # Video-Variablen initialisieren
    env.video_w = 640
    env.video_h = 352
    env.video_writer = None
    env._video_path = None
    env._frame_count = 0
    
    # Methoden als Instanz-Methoden hinzufügen
    recorder = VideoRecorder(env.video_w, env.video_h)
    
    # Methoden binden
    env.start_video = recorder.start_video
    env.capture_frame = lambda: recorder.capture_frame(env.renderer, env.data, env.cam, env.opt)
    env.finish_video = recorder.finish_video