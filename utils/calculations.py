import numpy as np

def calculate_center_vector(mu_start, mu_target):
    cx = 0
    for i in range(5):
        cx += mu_start[i][0]
    cx /= 5

    cy = 0
    for i in range(5):
        cy += mu_start[i][1]
    cy /= 5

    ctx = 0
    for i in range(5):
        ctx += mu_target[i][0]
    ctx /= 5

    cty = 0
    for i in range(5):
        cty += mu_target[i][1]
    cty /= 5

    return [ctx - cx, cty - cy], [cx, cy], [ctx, cty]

def calculate_center_point(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4):
    cx = (x0 + x1 + x2 + x3 + x4) / 5
    cy = (y0 + y1 + y2 + y3 + y4) / 5

    return [cx, cy]

def signierter_winkel_2d(v1, v2, in_grad=True):
    # optional: Nullvektoren abfangen
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        raise ValueError("Nullvektor hat keine definierte Richtung.")

    dot = np.dot(v1, v2)                       # Skalarprodukt
    det = v1[0] * v2[1] - v1[1] * v2[0]        # 2D-Determinante (z-Komponente vom Kreuzprodukt)

    winkel = np.arctan2(det, dot)              # signierter Winkel in Radiant [-pi, pi]

    if in_grad:
        winkel = np.degrees(winkel)

    return winkel

def calculate_rotations(mu_start, mu_target, vx, ct):
    v1 = np.asarray(vx, dtype=float)
    v2 = np.asarray(ct, dtype=float)

    mu_angle = []
    
    for i in range(5):
        v1 = np.asarray(mu_start[i], dtype=float)
        v2 = np.asarray(mu_target[i], dtype=float)
        angle = signierter_winkel_2d(v1 + vx - ct, v2 - ct)
        mu_angle.append(angle)
    
    return mu_angle

def calculate_rotations_t(
        x0, y0, x1, y1, x2, y2, x3, y3, x4, y4,
        xt0, yt0, xt1, yt1, xt2, yt2, xt3, yt3, xt4, yt4,
        vx, ct):
    
    v1 = np.asarray(vx, dtype=float)
    v2 = np.asarray(ct, dtype=float)

    mu_pos = [
        [x0, y0],
        [x1, y1],
        [x2, y2],
        [x3, y3],
        [x4, y4]
    ]

    mu_t = [
        [xt0, yt0],
        [xt1, yt1],
        [xt2, yt2],
        [xt3, yt3],
        [xt4, yt4]
    ]

    mu_angle = []
    
    for i in range(5):
        v1 = np.asarray(mu_pos[i], dtype=float)
        v2 = np.asarray(mu_t[i], dtype=float)
        angle = signierter_winkel_2d(v1 + vx - ct, v2 - ct)
        mu_angle.append(angle)
    
    return mu_angle

def calc_center_dists(
        mu_pos,
        cs):
    
    cs = np.asarray(cs, dtype=float)
    
    dists = []

    for i in range(5):
        vm = np.asarray(mu_pos[i], dtype=float)
        pm = vm - cs
        dists.append(np.linalg.norm(pm))

    return dists