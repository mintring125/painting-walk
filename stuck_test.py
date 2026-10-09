"""Count places where keyboard walking gets stuck although the ground ahead is walkable (same rules as main.js tryMove)."""
import json
import sys

import numpy as np

scene = json.load(open(sys.argv[1], encoding="utf-8"))
hf = scene["heightfield"]
H = np.array([np.nan if v is None else v for v in hf["h"]]).reshape(hf["nz"], hf["nx"])
c = hf["cell"]


def walk(x, z):
    i, j = int(round((z - hf["z0"]) / c)), int(round((x - hf["x0"]) / c))
    return 0 <= i < hf["nz"] and 0 <= j < hf["nx"] and np.isfinite(H[i, j])


def try_move(p, dx, dz):
    ok = lambda x, z: walk(x, z) and walk(x + dx * 2, z + dz * 2)
    for a in (0, 0.52, -0.52, 1.05, -1.05):
        cs, sn = np.cos(a), np.sin(a)
        mx, mz = dx * cs - dz * sn, dx * sn + dz * cs
        if ok(p[0] + mx, p[1] + mz):
            return p[0] + mx, p[1] + mz
    if walk(p[0] + dx, p[1]):
        return p[0] + dx, p[1]
    if walk(p[0], p[1] + dz):
        return p[0], p[1] + dz
    return p


from scipy import ndimage
lab, _ = ndimage.label(~np.isfinite(H))
sizes = np.bincount(lab.ravel())
border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist())


def blocker_is_speck(x, z):
    i, j = int(round((z - hf["z0"]) / c)), int(round((x - hf["x0"]) / c))
    if not (0 <= i < hf["nz"] and 0 <= j < hf["nx"]) or np.isfinite(H[i, j]):
        return False
    k = lab[i, j]
    return k not in border and sizes[k] <= 60


iz, ix = np.nonzero(np.isfinite(H))
rng = np.random.default_rng(1)
pick = rng.choice(len(iz), min(300, len(iz)), replace=False)
stuck, specks, spots = 0, 0, []
for k in pick:
    start = (ix[k] * c + hf["x0"], iz[k] * c + hf["z0"])
    for ang in np.arange(0, 2 * np.pi, np.pi / 4):
        d = (np.sin(ang) * 0.04, np.cos(ang) * 0.04)
        p = start
        for _ in range(40):  # 1.6 m
            q = try_move(p, *d)
            if q == p:
                # stuck: is there walkable ground 0.5-1.0 m further on? then it's a speck, not a wall
                if any(walk(p[0] + d[0] * s, p[1] + d[1] * s) for s in (12, 18, 25)):
                    stuck += 1
                    if any(blocker_is_speck(p[0] + d[0] * s_, p[1] + d[1] * s_) for s_ in (1, 2, 3)):
                        specks += 1
                        spots.append((round(float(p[0]), 1), round(float(p[1]), 1)))
                break
            p = q
print(f"{sys.argv[1]}: stuck {stuck} of {len(pick) * 8} walks, {specks} on small specks", spots[:6])
