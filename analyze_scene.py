"""Turn a SHARP single-image splat into a walkable, interactive scene description.

Outputs scene.json with:
  - splat -> world transform (world: Y up, ground under the painting camera at y=0)
  - the painting camera (position, orientation, vertical FOV, aspect)
  - a walkable heightfield built from the splat points (stands in for a photogrammetry collision mesh)
  - hotspots found from pixel boxes in the source image: sit (seat top), touch (walk up and reach), look (turn and gaze)

Hotspots come from a JSON list (--hotspots file):
  [{"id": "bench", "label": "벤치", "action": "sit", "box": [u0, v0, u1, v1],
    "facing": "camera" | [x, z], "approach": 0.5, "say": "..."}, ...]
A bare positional seat box is still accepted and becomes one "sit" hotspot.
"""
import argparse
import json

import numpy as np
from plyfile import PlyData

ap = argparse.ArgumentParser()
ap.add_argument("ply")
ap.add_argument("out")
ap.add_argument("seat", nargs="?", help="u0,v0,u1,v1 pixel box of a seat top in the source image")
ap.add_argument("--hotspots", help="JSON file with the hotspot list")
ap.add_argument("--facing", default="camera",
                help="default seated facing: 'camera' (toward the viewer), or world 'x,z'")
ap.add_argument("--max-step", type=float, default=0.06,
                help="largest height change between 10 cm cells that still counts as walkable (stairs: ~0.22)")
ap.add_argument("--approach", type=float, default=0.5, help="metres in front of a seat where the character stops")
ap.add_argument("--ground-ceiling", type=float,
                help="ignore ground higher than this (m); default just below the camera, raise it for stairs/slopes going up")
ap.add_argument("--stairs-top", help="u0,v0,u1,v1 pixel box of the top landing of a long stairway; the stairs are "
                "rebuilt as a continuous walkable ramp of treads up to it (far splats are too sparse to walk on)")
ap.add_argument("--stairs-from", type=float, default=-25.0, help="world z where the rebuilt stairway takes over")
ap.add_argument("--stairs-halfwidth", type=float, default=2.0, help="half width of the rebuilt stairway (m)")
ap.add_argument("--landing", type=float, default=4.0, help="flat landing added beyond the top (m)")
ap.add_argument("--corridor-free", action="store_true",
                help="the rebuilt path may go down as well as up (roads over a crest); default keeps rising (stairs)")
ap.add_argument("--corridor-drift", type=float, default=0.6,
                help="how far the path centre follows the end point sideways (1 = all the way)")
ap.add_argument("--corridor-pct", type=float, default=60,
                help="height percentile per depth bin of the rebuilt path: 60 = tread tops; ~20 on wet, reflective ground")
ap.add_argument("--fill-holes", type=int, default=60,
                help="fill enclosed non-walkable specks up to this many cells; person-height things (poles) are kept")
ap.add_argument("--no-obstacles", action="store_true",
                help="keep cells walkable even when something occupies the body's space above the floor (tables, benches)")
ap.add_argument("--exclude-green", action="store_true",
                help="drop ground cells whose surface is clearly green (bushes, hillside grass)")
ap.add_argument("--eye-height", type=float,
                help="rescale the scene so the painting camera stands this high (m); paintings rarely give SHARP true scale")
args = ap.parse_args()

ply = PlyData.read(args.ply)
v = ply["vertex"]
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
opacity = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
rgb = np.clip(0.5 + 0.28209479 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), 0, 1)
K = np.asarray(ply["intrinsic"]["intrinsic"], np.float64).reshape(3, 3)
W, H = [int(s) for s in ply["image_size"]["image_size"]]
print(f"{len(xyz)} splats, image {W}x{H}, fx={K[0,0]:.1f} fy={K[1,1]:.1f}")

# Project every splat into the source image (OpenCV camera: x right, y down, z forward).
z = xyz[:, 2]
u = K[0, 0] * xyz[:, 0] / z + K[0, 2]
w = K[1, 1] * xyz[:, 1] / z + K[1, 2]
solid = (opacity > 0.5) & (z > 0.05)

# --- Ground plane: RANSAC over the floor at the bottom-centre of the picture.
road = solid & (w > 0.78 * H) & (np.abs(u - W / 2) < 0.30 * W)
pts = xyz[road]
rng = np.random.default_rng(0)
best, best_n = None, 0
for _ in range(400):
    a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
    n = np.cross(b - a, c - a)
    if np.linalg.norm(n) < 1e-9:
        continue
    n /= np.linalg.norm(n)
    inl = np.abs((pts - a) @ n) < 0.02
    if inl.sum() > best_n:
        best, best_n = inl, inl.sum()
P = pts[best]
centroid = P.mean(0)
n = np.linalg.svd(P - centroid, full_matrices=False)[2][2]
if n[1] > 0:  # make the normal point up (OpenCV y is down)
    n = -n
cam_height = abs(centroid @ n)  # camera sits at the origin
scale = args.eye_height / cam_height if args.eye_height else 1.0
print(f"ground: {best_n}/{len(pts)} inliers, camera height {cam_height:.2f} m, scale x{scale:.2f}")
cam_height *= scale

# --- World frame: Y = up (plane normal), -Z = camera forward projected on the ground.
up = n
fwd_cv = np.array([0.0, 0.0, 1.0])
fwd = fwd_cv - (fwd_cv @ up) * up
fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, up)  # three.js: x right, y up, z backwards
R = np.stack([right, up, -fwd])  # rows map cv vectors into world axes
t = np.array([0.0, cam_height, 0.0])  # camera position in world
M = np.eye(4)
M[:3, :3] = R * scale
M[:3, 3] = t
world = (xyz @ R.T) * scale + t

cam_rot = R @ np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes -> three.js camera axes
fov_y = 2 * np.degrees(np.arctan(H / 2 / K[1, 1]))

# --- Walkable heightfield (the "photogrammetry" collision surface).
cell = 0.10
ceiling = args.ground_ceiling if args.ground_ceiling is not None else cam_height - 0.2
g = solid & (world[:, 1] < ceiling)
gx, gz, gy = world[g, 0], world[g, 2], world[g, 1]
x0, x1 = np.percentile(gx, [1, 99])
z0, z1 = np.percentile(gz, [1, 99])
z0 = max(z0, -60.0)
if args.stairs_top:
    # make room for a rebuilt path that reaches beyond the dense ground
    _u0, _v0, _u1, _v1 = (int(x_) for x_ in args.stairs_top.split(","))
    _tb = solid & (u >= _u0) & (u <= _u1) & (w >= _v0) & (w <= _v1)
    _top = np.median(world[_tb], 0)
    z0 = min(z0, _top[2] - args.landing - 1.0)
    x0 = min(x0, _top[0] - args.stairs_halfwidth - 1.0)
    x1 = max(x1, _top[0] + args.stairs_halfwidth + 1.0)
nx, nz = int((x1 - x0) / cell) + 1, int((z1 - z0) / cell) + 1
ix = ((gx - x0) / cell).astype(int)
iz = ((gz - z0) / cell).astype(int)
ok = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
flat = iz[ok] * nx + ix[ok]
order = np.lexsort((gy[ok], flat))
flat_s, y_s = flat[order], gy[ok][order]
starts = np.r_[0, np.flatnonzero(np.diff(flat_s)) + 1]
counts = np.diff(np.r_[starts, len(flat_s)])
height = np.full(nx * nz, np.nan)
# lowest well-supported surface in each cell = ground under walls, poles and plants
low = y_s[starts + np.minimum(counts - 1, counts // 10)]
height[flat_s[starts[counts >= 3]]] = low[counts >= 3]
tall_cells = np.zeros(nx * nz, bool)
if not args.no_obstacles:
    # anything solid between knee-ish and head height above a cell's floor blocks it (low tables, benches, pedestals)
    sw = world[solid]
    oi = ((sw[:, 0] - x0) / cell).astype(int)
    oj = ((sw[:, 2] - z0) / cell).astype(int)
    inside = (oi >= 0) & (oi < nx) & (oj >= 0) & (oj < nz)
    of = oj[inside] * nx + oi[inside]
    rel = sw[inside, 1] - height[of]
    low_band = max(0.15, args.max_step * 1.4)  # a climbable step riser is not an obstacle
    occ = np.bincount(of[(rel > low_band) & (rel < 1.5)], minlength=nx * nz)
    tall_cells = np.bincount(of[(rel > 1.0) & (rel < 2.2)], minlength=nx * nz) >= 4
    blocked = (occ >= 6) & np.isfinite(height)
    height[blocked] = np.nan
    print(f"blocked {blocked.sum()} cells with obstacles")
if args.exclude_green:
    # colour of the lowest third of each cell = the walking surface
    col = rgb[g][ok][order]
    rank = np.arange(len(flat_s)) - np.repeat(starts, counts)
    keep = rank <= np.repeat(counts, counts) // 3
    sums = np.zeros((nx * nz, 3))
    np.add.at(sums, flat_s[keep], col[keep])
    num = np.bincount(flat_s[keep], minlength=nx * nz)[:, None]
    mean = sums / np.maximum(num, 1)
    green = (mean[:, 1] > mean[:, 0] * 1.08) & (mean[:, 1] > mean[:, 2] * 1.05) & (num[:, 0] > 0)
    height[green] = np.nan
    print(f"excluded {green.sum()} green cells")
height = height.reshape(nz, nx)

# Far away the splats thin out and leave holes; fill a hole from its neighbours when most of them are ground.
for _ in range(3):
    pad = np.pad(height, 1, constant_values=np.nan)
    nb = np.stack([pad[1 + dz:nz + 1 + dz, 1 + dx:nx + 1 + dx]
                   for dz in (-1, 0, 1) for dx in (-1, 0, 1) if dz or dx])
    valid = np.isfinite(nb).sum(0)
    fill = ~np.isfinite(height) & (valid >= 5)
    if not args.no_obstacles:
        fill &= ~blocked.reshape(nz, nx)
    height[fill] = np.nanmedian(nb[:, fill], axis=0)

if args.stairs_top:
    # Rebuild a long stairway from its centre-line profile: robust tread height per 20 cm of depth, kept rising.
    tu0, tv0, tu1, tv1 = (int(x_) for x_ in args.stairs_top.split(","))
    tb = solid & (u >= tu0) & (u <= tu1) & (w >= tv0) & (w <= tv1)
    top = np.median(world[tb], 0)
    top[1] = np.percentile(world[tb][:, 1], 75 if args.corridor_pct >= 50 else args.corridor_pct)  # visible surface
    zs_from, zs_top, hw = args.stairs_from, float(top[2]), args.stairs_halfwidth
    cx_at = lambda zz: top[0] * args.corridor_drift * (zz - zs_from) / (zs_top - zs_from)  # drift toward the end
    bins = np.arange(zs_from, zs_top - 0.2, -0.2)
    prof = np.full(len(bins), np.nan)
    for i, b0 in enumerate(bins):
        mm = solid & (world[:, 2] <= b0) & (world[:, 2] > b0 - 0.2) & (np.abs(world[:, 0] - cx_at(b0)) < hw * 0.7)             & (world[:, 1] < top[1] + 0.3)
        if mm.sum() >= 6:
            prof[i] = np.percentile(world[mm, 1], args.corridor_pct)  # tread tops; lower values sink into steps
    # start from the real heightfield where the rebuild takes over, end on the landing
    ci = int(round((zs_from - z0) / cell))
    cj = int(round((cx_at(zs_from) - x0) / cell))
    start_y = np.nanmedian(height[max(0, ci - 2):ci + 3, max(0, cj - 5):cj + 6])
    prof[0], prof[-1] = start_y, top[1]
    good = np.isfinite(prof)
    prof = np.interp(np.arange(len(bins)), np.flatnonzero(good), prof[good])
    if args.corridor_free:
        k_ = 7  # running median: smooth, but free to dip over a crest
        pad_ = np.pad(prof, k_ // 2, mode="edge")
        prof = np.array([np.median(pad_[i_:i_ + k_]) for i_ in range(len(prof))])
    else:
        prof = np.maximum.accumulate(prof)
        prof = np.minimum(prof, top[1])
    for i, b0 in enumerate(bins):
        zi = int(round((b0 - z0) / cell))
        for zz in range(zi - 1, zi + 2):
            if not 0 <= zz < nz:
                continue
            c0 = int(round((cx_at(b0) - hw - x0) / cell)); c1 = int(round((cx_at(b0) + hw - x0) / cell))
            height[zz, max(0, c0):min(nx, c1 + 1)] = prof[i]
    # flat landing beyond the top step
    for zz in range(max(0, int(round((zs_top - args.landing - z0) / cell))), min(nz, int(round((zs_top - z0) / cell)) + 1)):
        c0 = int(round((cx_at(zs_top) - hw - x0) / cell)); c1 = int(round((cx_at(zs_top) + hw - x0) / cell))
        height[zz, max(0, c0):min(nx, c1 + 1)] = top[1]
    print(f"stairway rebuilt from z {zs_from:.1f} to top landing z {zs_top:.1f} (y {top[1]:.2f})")

# Flood fill from just past the picture's bottom edge; walkable = continuous, gentle steps.
spawn_z = -cam_height / np.tan(np.radians(fov_y / 2)) - 1.0
fz, fx = np.nonzero(np.isfinite(height))
k = np.argmin((fz * cell + z0 - spawn_z) ** 2 + (fx * cell + x0) ** 2)
seed = (int(fz[k]), int(fx[k]))
walk = np.zeros_like(height, bool)
walk[seed] = True
stack = [seed]
while stack:
    cz, cx = stack.pop()
    h = height[cz, cx]
    for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        qz, qx = cz + dz, cx + dx
        if 0 <= qz < nz and 0 <= qx < nx and not walk[qz, qx]:
            q = height[qz, qx]
            if np.isfinite(q) and abs(q - h) < args.max_step:
                walk[qz, qx] = True
                stack.append((qz, qx))
if args.fill_holes:
    # Clean the walkable mask: single-image splats leave specks and thin cracks that stop the character.
    from scipy import ndimage
    tall = tall_cells.reshape(nz, nx)
    before = walk.sum()
    # 1) close cracks up to ~2 cells wide (never across person-height obstacles)
    closed = ndimage.binary_closing(walk, structure=np.ones((3, 3)), iterations=1) & ~tall
    # 2) fill enclosed specks that are small and low
    lab, n_ = ndimage.label(~closed)
    sizes = np.bincount(lab.ravel())
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist())
    for k_ in range(1, n_ + 1):
        if k_ in border or sizes[k_] > args.fill_holes:
            continue
        hole = lab == k_
        if tall[hole].mean() > 0.3:
            continue
        closed |= hole
    new = closed & ~walk
    # heights for new cells: median of walkable neighbours, grown outward a few rings
    hgt = np.where(walk, height, np.nan)
    for _ in range(6):
        todo = new & ~np.isfinite(hgt)
        if not todo.any():
            break
        pad = np.pad(hgt, 1, constant_values=np.nan)
        nb = np.stack([pad[1 + dz:nz + 1 + dz, 1 + dx:nx + 1 + dx] for dz in (-1, 0, 1) for dx in (-1, 0, 1) if dz or dx])
        cnt = np.isfinite(nb).sum(0)
        grow = todo & (cnt > 0)
        hgt[grow] = np.nanmedian(nb[:, grow], axis=0)
    new &= np.isfinite(hgt)
    height[new] = hgt[new]
    walk |= new
    print(f"cleanup: +{walk.sum() - before} cells (cracks and specks)")
hf = np.where(walk, height, np.nan)
wz, wx = np.nonzero(walk)
walk_xz = np.stack([wx * cell + x0, wz * cell + z0], 1)
print(f"heightfield {nx}x{nz} cells, walkable {walk.sum()} ({walk.sum() * cell * cell:.1f} m^2)")


def floor_near(x, zz):
    ci, cj = int(round((zz - z0) / cell)), int(round((x - x0) / cell))
    win = hf[max(0, ci - 3):ci + 4, max(0, cj - 3):cj + 4]
    return float(np.nanmedian(win)) if np.isfinite(win).any() else None


def parse_facing(f, c):
    if f in (None, "camera"):
        d = np.array([-c[0], 0.0, -c[2]])  # toward the camera above the world origin
    else:
        fx_, fz_ = (float(x) for x in (f.split(",") if isinstance(f, str) else f))
        d = np.array([fx_, 0.0, fz_])
    return d / np.linalg.norm(d)


def seat_edge(c, face, seat_y):
    """Distance from the seat point to the cushion's front edge, from all splats around the seat (not just the box)."""
    rel = world - c
    along = rel @ face
    side = rel @ np.cross(np.array([0.0, 1.0, 0.0]), face)
    m = solid & (np.abs(side) < 0.25) & (world[:, 1] > seat_y - 0.12) & (world[:, 1] < seat_y + 0.04)         & (along > -0.3) & (along < 0.9)
    if m.sum() < 20:
        return 0.0
    # walk forward from the seat point and stop at the first gap: tables in front of a sofa are not the cushion
    a_ = np.sort(along[m][along[m] >= 0])
    if len(a_) == 0:
        return 0.0
    gaps = np.flatnonzero(np.diff(a_) > 0.06)
    return float(min(a_[gaps[0]] if len(gaps) else a_[-1], 0.7))


def seat_surface(c, face, floor_y):
    """Find the cushion: the longest flat run of the top surface along the facing direction.
    Returns (seat height, front edge distance from c) or None. Boxes often catch the backrest or the cushion's back."""
    rel = world - c
    along = rel @ face
    side = rel @ np.cross(np.array([0.0, 1.0, 0.0]), face)
    m = solid & (np.abs(side) < 0.25) & (rel[:, 1] > floor_y - c[1] + 0.15) & (rel[:, 1] < 0.4)
    bins = np.arange(-0.15, 0.9, 0.05)
    top = np.full(len(bins), np.nan)
    for i_, a_ in enumerate(bins):
        mm = m & (along >= a_) & (along < a_ + 0.05)
        if mm.sum() >= 5:
            top[i_] = np.percentile(rel[mm, 1], 90)
    best, run = None, []
    for i_ in range(len(bins)):
        if np.isfinite(top[i_]) and run and abs(top[i_] - np.median(top[run])) < 0.04:
            run.append(i_)
        else:
            if run and (best is None or len(run) > len(best)):
                best = run
            run = [i_] if np.isfinite(top[i_]) else []
    if run and (best is None or len(run) > len(best)):
        best = run
    if not best or len(best) < 4 or bins[best[0]] > 0.15:
        return None  # the flat run must start at the seat point, not be a step or the floor further on
    y_ = float(c[1] + np.median(top[best]))
    if abs(y_ - c[1]) > 0.2:
        return None  # a backrest or a lower step, not this cushion
    return y_, float(min(bins[best[-1]] + 0.05, 1.0))


# --- Hotspots.
spec = json.load(open(args.hotspots, encoding="utf-8")) if args.hotspots else []
if args.seat:
    spec.insert(0, {"id": "seat", "label": "의자", "action": "sit",
                    "box": [int(x) for x in args.seat.split(",")], "facing": args.facing})
hotspots = []
for h in spec:
    u0, v0, u1, v1 = h["box"]
    inb = solid & (u >= u0) & (u <= u1) & (w >= v0) & (w <= v1)
    bp, depth = world[inb], z[inb]
    if len(bp) < 5:
        print(f"WARNING: hotspot {h['id']}: no splats in box, skipped")
        continue
    out = {k_: h[k_] for k_ in ("id", "label", "action") if k_ in h}
    out["say"] = h.get("say", "")
    # pick radius: half the box width at the box's depth, so far objects stay clickable
    out["radius"] = float(max(0.3, (u1 - u0) / K[0, 0] * np.median(depth) / 2))
    if h["action"] == "npc":
        # a background character: seated on a seat top, or standing on the floor in the box
        for k_ in ("pose", "idle", "greet", "lines", "model", "role", "tint"):
            if k_ in h:
                out[k_] = h[k_]
        if h.get("pose") == "sit":
            seat_y = np.percentile(bp[:, 1], 90)
            top = bp[bp[:, 1] > seat_y - 0.05]
            c = np.median(top, 0)
            c[1] = seat_y
            f_ = parse_facing(h.get("facing", "camera"), c)
            out["edge"] = float(h["edge"]) if "edge" in h else float(np.clip(seat_edge(c, f_, seat_y), 0.05, 0.35))
            gf = floor_near(c[0] + f_[0] * 0.8, c[2] + f_[2] * 0.8)
            surf = seat_surface(c, f_, gf if gf is not None else seat_y - 0.45)
            if surf and "edge" not in h:
                c[1], out["edge"] = surf
        else:
            near = bp[depth <= np.percentile(depth, 30)]  # the visible floor in front, not what is behind it
            c = np.median(near, 0)
            c[1] = np.percentile(near[:, 1], 20)
        face = parse_facing(h.get("facing", "camera"), c)
        out.update(target=c.tolist(), at=c.tolist(), facing=face.tolist())
        print(f"npc   {h['id']:<10} {h.get('pose', 'stand')} at {np.round(c, 2)}"
              + (f", front edge {out['edge']:.2f} m ahead" if "edge" in out else ""))
        hotspots.append(out)
        continue
    if h["action"] == "sit":
        seat_y = np.percentile(bp[:, 1], 90)
        top = bp[bp[:, 1] > seat_y - 0.05]
        c = np.median(top, 0)
        face = parse_facing(h.get("facing", args.facing), c)
        edge = float(h["edge"]) if "edge" in h else float(np.clip(seat_edge(c, face, seat_y), 0.05, 0.35))
        guess_floor = floor_near(c[0] + face[0] * 0.8, c[2] + face[2] * 0.8)
        surf = seat_surface(c, face, guess_floor if guess_floor is not None else seat_y - 0.45)
        if surf and "edge" not in h:
            seat_y, edge = surf
        approach = max(float(h.get("approach", args.approach)), edge + 0.3)  # stop clear of the seat itself
        floor = floor_near(c[0] + face[0] * approach, c[2] + face[2] * approach)
        if floor is None:
            print(f"WARNING: hotspot {h['id']}: no walkable ground in front of the seat; try another facing")
            floor = float(np.nanmedian(hf))
        out.update(target=[float(c[0]), float(seat_y), float(c[2])], seat=[float(c[0]), float(seat_y), float(c[2])],
                   facing=face.tolist(), approach=approach, floorY=floor, seatHeight=float(seat_y - floor), edge=edge)
        print(f"sit   {h['id']:<10} seat {np.round(c, 2)} height {seat_y - floor:.2f} m, front edge {edge:.2f} m ahead")
    else:
        c = np.median(bp, 0)
        out["target"] = c.tolist()
        if h["action"] in ("touch", "pray"):
            # touch: stand an arm's length away (camera side preferred); pray: the closest walkable spot, however far
            d = np.hypot(walk_xz[:, 0] - c[0], walk_xz[:, 1] - c[2])
            cand = np.flatnonzero((d > 0.3) & (d < 1.2)) if h["action"] == "touch" else np.array([], int)
            if len(cand) == 0:
                cand = np.argsort(d)[:1]
            cam_d = np.hypot(walk_xz[cand, 0], walk_xz[cand, 1])
            pick = cand[np.argmin(np.abs(d[cand] - 0.55) + 0.02 * cam_d)]
            ax, az = walk_xz[pick]
            face = np.array([c[0] - ax, 0.0, c[2] - az])
            face /= np.linalg.norm(face)
            out.update(stand=[float(ax), float(hf[wz[pick], wx[pick]]), float(az)], facing=face.tolist(),
                       reach=float(d[pick]))
            print(f"{h['action']:<5} {h['id']:<10} target {np.round(c, 2)} stand {np.round([ax, az], 2)} reach {d[pick]:.2f} m")
        else:
            print(f"look  {h['id']:<10} target {np.round(c, 2)}")
    hotspots.append(out)

sits = [h for h in hotspots if h["action"] == "sit"]
result = {
    "splatToWorld": M.T.flatten().tolist(),  # column-major for THREE.Matrix4.fromArray
    "camera": {"position": t.tolist(), "rotation": cam_rot.T.flatten().tolist(), "fovY": fov_y, "aspect": W / H},
    "heightfield": {"x0": float(x0), "z0": float(z0), "cell": cell, "nx": nx, "nz": nz,
                    "h": [None if not np.isfinite(x) else round(float(x), 3) for x in hf.ravel()]},
    "hotspots": hotspots,
    "bench": sits[0] if sits else None,  # first seat, for older viewers
    "spawn": [float(seed[1] * cell + x0), float(height[seed]), float(seed[0] * cell + z0)],
}
json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
print("wrote", args.out)
