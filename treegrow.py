"""Trees grown, then drawn: a skeleton first, leaves hung on it, pixels last.

How a tree grows decides what it looks like, so each species here is a way of growing, not a
set of sizes:

- Apical dominance. Where the leading shoot keeps the side shoots short (excurrent: pine, alder,
  birch) there is one straight stem to the top and a cone of branches round it. Where it soon
  loses control (decurrent: oak, willow) the trunk forks low into a few big limbs that spread
  out and up into a round or broad crown.
- Branch angle. A branch holds a set angle to the vertical (its gravitropic set point): steep
  in oak and alder, near level in pine, which droops with age; birch and willow twigs hang.
- Where the leaves are. An oak in the open carries its leaves at the ends of its limbs, in a
  shell round an inside you can see branches through; a pine carries its needles in flat pads
  at the ends of level branches; a birch's crown is open and its fine twigs hang.
- Self-pruning. Shaded lower branches die: an old pine stands on a long bare stem under a
  flat top; an alder in a wood keeps a short crown.
- Wind. On an exposed coast the branches on the windward side are killed, and the tree is a
  flag pointing downwind.
- Thickness. Leonardo's rule: the limbs above a fork together are about as thick as the stem
  below it (cross-sections add up; exponent ~2).

The skeleton is grown by space colonisation: the crown's room is filled with attraction points
and each branch tip grows toward the points nearest it, using them up as it reaches them. The
room's shape, how its points clump, the leader and the set angle are the species. Pixels last:
bark lit from the left along each limb, leaves gathered into a few lobes shaded as balls inside
one bigger ball, five tones.

Sources: Hallé, Oldeman & Tomlinson, Tropical Trees and Forests (architectural models);
Alabama Extension, How Tree Growth Affects Tree Failure (excurrent/decurrent); Oregon State
Landscape Plants, Pinus sylvestris; Trees and Shrubs Online, Betula pendula; RHS, Alnus
glutinosa; Minamino & Tateno 2014, Tree branching: Leonardo da Vinci's rule versus
biomechanical models; Roychoudhry et al. 2013 (gravitropic set-point angle); Krummholz and
flag trees (Wikipedia).

A prototype, not yet used by the tileset.
"""
from __future__ import annotations

import math
import random

import numpy as np

from trees import _clean, _hash, _image, _shift

T = 0
L = [1, 2, 3, 4, 5]          # leaf tones, deepest .. lightest
B = [6, 7, 8, 9]              # bark, darkest .. lit
U = [10, 11, 12, 13]          # bark high up (a Scots pine's orange top), same order
# where the light on the leaves comes from: above and a little to the left, so the top of a crown
# is its brightest part and a lobe out on its left side is lit, but not as a pale ball of its own
LX, LY = 0.42, 0.91

LEAF = {   # the foliage ramp (l): cooler and deeper than the grass, so a crown stands off the ground
    "oak": ["l0", "l1", "l2", "l3", "l4"],
    "alder": ["b0", "l0", "l1", "l2", "l3"],
    "birch": ["l1", "l2", "l3", "l4", "l5"],
    "pine": ["b0", "b1", "l0", "l1", "l2"],
    "willow": ["l0", "l1", "l2", "t0", "t1"],
}
BARK = {
    "oak": ["n0", "w0", "w1", "w2"],
    "alder": ["n0", "s0", "s1", "s2"],
    "birch": ["n0", "s3", "s4", "f0"],
    "pine": ["n0", "w0", "w1", "w2"],
    "willow": ["n0", "w0", "w1", "k0"],
}
BARK_UP = {"pine": ["n0", "w1", "o0", "o1"]}


# ---------------------------------------------------------------- skeleton

class Skeleton:
    def __init__(self):
        self.pos: list[tuple[float, float]] = []
        self.parent: list[int] = []

    def add(self, x, y, parent):
        self.pos.append((float(x), float(y)))
        self.parent.append(parent)
        return len(self.pos) - 1

    def children(self):
        ch = [[] for _ in self.pos]
        for i, p in enumerate(self.parent):
            if p >= 0:
                ch[p].append(i)
        return ch

    def radii(self, base, tip=0.45, exp=2.0):
        """Leonardo's rule from the tips down, then scaled so the stem's base is `base` px and
        a twig about one pixel."""
        ch = self.children()
        r = np.zeros(len(self.pos))
        for i in reversed(range(len(self.pos))):          # children always come after parents
            r[i] = 1.0 if not ch[i] else sum(r[c] ** exp for c in ch[i]) ** (1 / exp)
        top = max(r[i] for i, p in enumerate(self.parent) if p < 0)
        return tip + (r - 1.0) / max(top - 1.0, 1e-6) * (base - tip)


def _rot(v, deg):
    a = math.radians(deg)
    return np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)])


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def colonise(sk, pts, rng, step=1.4, reach=13.0, kill=2.4, pull=(0.0, 0.0), wobble=0.1,
             set_angle=None, set_k=0.0, start=None, iters=220):
    """Grow `sk` into the attraction points `pts`. set_angle: the angle from straight up (deg)
    that branches turn toward, with strength set_k. start: the nodes allowed to sprout (others
    stay bare)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    alive = np.ones(len(pts), dtype=bool)
    frozen = set() if start is None else set(range(len(sk.pos))) - set(start)
    first, reach0 = True, reach
    for _ in range(iters):
        if not alive.any():
            break
        P = np.array(sk.pos)
        A = pts[alive]
        d = np.hypot(A[:, None, 0] - P[None, :, 0], A[:, None, 1] - P[None, :, 1])
        if frozen:
            d[:, sorted(frozen)] = np.inf
        near = d.argmin(axis=1)
        ok = d[np.arange(len(A)), near] < reach
        if not ok.any():
            if first and reach < 60:                       # the crown starts further off: reach for it
                reach *= 1.3
                continue
            break
        first = False
        grow = {}
        for a, n in zip(np.nonzero(ok)[0], near[ok]):
            grow.setdefault(int(n), []).append(_unit(A[a] - P[n]))
        added = 0
        for n, vs in grow.items():
            v = _unit(np.sum(vs, axis=0))
            if set_angle is not None:
                side = 1.0 if v[0] >= 0 else -1.0
                a = math.radians(set_angle)
                v = _unit(v + set_k * np.array([side * math.sin(a), -math.cos(a)]))
            v = _unit(v + np.array(pull) + np.array([rng.uniform(-wobble, wobble), rng.uniform(-wobble, wobble)]))
            for turn in (0, 35, -35, 70, -70):              # pulled back the way it came: turn aside
                nx, ny = P[n] + _rot(v, turn) * step
                if np.min(np.hypot(P[:, 0] - nx, P[:, 1] - ny)) >= step * 0.5:
                    break
            else:
                continue
            sk.add(nx, ny, n)
            added += 1
        reach = max(reach0, reach - step)                 # (a long first reach shortens as it closes)
        idx = np.nonzero(alive)[0]
        P = np.array(sk.pos)
        dk = np.hypot(pts[idx, None, 0] - P[None, :, 0], pts[idx, None, 1] - P[None, :, 1]).min(axis=1)
        alive[idx[dk < kill]] = False
        if not added:
            break
    return sk


def stem(sk, x, y, to_y, rng, step=1.4, wobble=0.08, parent=-1, lean=0.0):
    """A straight-ish stem up to row `to_y`; returns the node ids, bottom first."""
    ids = [sk.add(x, y, parent)]
    while y > to_y:
        x += rng.uniform(-wobble, wobble) + lean * step
        y -= step
        ids.append(sk.add(x, y, ids[-1]))
    return ids


def limb(sk, from_id, direction, length, rng, step=1.4, bend=0.0, droop=0.0, wobble=0.12):
    """A branch from node `from_id`: `direction` turns by `bend` deg a step, and sags by `droop`."""
    x, y = sk.pos[from_id]
    v = _unit(np.array(direction, dtype=float))
    ids, n = [from_id], max(1, int(length / step))
    for k in range(n):
        v = _unit(_rot(v, bend) + np.array([rng.uniform(-wobble, wobble), droop * k / n]))
        x, y = x + v[0] * step, y + v[1] * step
        ids.append(sk.add(x, y, ids[-1]))
    return ids[1:]


# ---------------------------------------------------------------- rooms (where the crown may go)

def room(rng, cx, cy, rx, ry, clumps=(7, 10), per=(25, 45), under=0.7, shell=0.0, upper_only=0.8, keep=None):
    """Attraction points in an ellipse (flatter underneath by `under`), laid in clumps so the
    crown has gaps. shell > 0 keeps only an outer shell that thick (as a share of the radius).
    keep: an extra test (x, y) -> bool."""
    pts = []
    for _ in range(rng.randint(*clumps)):
        if rng.random() < upper_only:
            a = rng.uniform(math.pi * 1.05, math.pi * 1.95)
        else:
            a = rng.uniform(0, math.pi * 2)
        d = rng.uniform(0.35, 0.85)
        kx, ky = cx + math.cos(a) * rx * d, cy + math.sin(a) * ry * d * 0.9
        kr = rng.uniform(0.25, 0.4) * rx
        for _ in range(rng.randint(*per)):
            px, py = kx + rng.gauss(0, kr * 0.55), ky + rng.gauss(0, kr * 0.45)
            ny = (py - cy) / (ry if py < cy else ry * under)
            e = ((px - cx) / rx) ** 2 + ny ** 2
            if (shell <= 0 or (1 - shell) ** 2 < e) and e < 1 and (keep is None or keep(px, py)):
                pts.append((px, py))
    return np.array(pts).reshape(-1, 2)


def cone(rng, cx, top, bottom, half, clumps=(9, 12), per=(14, 24), keep=None):
    """Attraction points in a cone, laid in clumps (each a branch's spray) so it has gaps."""
    def inside(x, y):
        return top <= y <= bottom and abs(x - cx) <= half * ((y - top) / (bottom - top)) ** 0.9
    pts = []
    for _ in range(rng.randint(*clumps)):
        while True:
            ky = top + (bottom - top) * rng.random() ** 0.7
            kx = cx + rng.uniform(-1, 1) * half * ((ky - top) / (bottom - top)) ** 0.9
            if inside(kx, ky):
                break
        kr = half * rng.uniform(0.18, 0.28)
        for _ in range(rng.randint(*per)):
            x, y = kx + rng.gauss(0, kr), ky + rng.gauss(0, kr * 0.6)
            if inside(x, y) and (keep is None or keep(x, y)):
                pts.append((x, y))
    return np.array(pts).reshape(-1, 2)


# ---------------------------------------------------------------- species: how each one grows

def grow_oak(rng, w, h):
    """Decurrent: a short stem forks low into 2-4 big limbs that climb steeply and spread; the
    leaves sit in a shell at the limb ends (open-grown: an almost perfect hemisphere)."""
    sk = Skeleton()
    base = stem(sk, w / 2 - 0.5 + rng.uniform(-1, 1), h - 2, h * rng.uniform(0.62, 0.7), rng)
    pts = room(rng, w / 2, h * 0.42, w * 0.47, h * 0.37, under=0.75, shell=0.7)
    n = rng.randint(2, 4)                                  # the limbs: each heads for its own part
    for k in range(n):
        a = -55 + 110 * (k + 0.5) / n + rng.uniform(-12, 12)
        limb(sk, base[-1], _rot(np.array([0, -1.0]), a), rng.uniform(7, 12), rng, bend=rng.uniform(-4, 4))
    colonise(sk, pts, rng, set_angle=40, set_k=0.35, wobble=0.22, start=range(len(base), len(sk.pos)))
    return sk, dict(base=3.2, leaves="clumps", clump=(3.6, 5.2), lobes=(6, 8), fine=1.2, low=0.72)


def grow_alder(rng, w, h, wood=False):
    """Excurrent: one stem nearly to the top, branches at a steep set angle; a cone. In a wood
    the shaded lower branches are gone."""
    sk = Skeleton()
    top = h * 0.06
    lead = stem(sk, w / 2 - 0.5, h - 2, top, rng, wobble=0.05)
    low = h * (0.5 if wood else 0.68)
    pts = cone(rng, w / 2, top + 2, low, w * 0.46)
    colonise(sk, pts, rng, set_angle=55, set_k=0.4, wobble=0.12, reach=10,
             start=[i for i in lead if sk.pos[i][1] < low])
    return sk, dict(base=2.3, leaves="clumps", clump=(2.3, 3.3), lobes=(8, 11), fine=0.85, low=low / h + 0.02)


def grow_birch(rng, w, h):
    """Excurrent and slender: a white stem nearly to the top, few light branches at a steep
    angle, an open, narrow, uneven crown, and long thin twigs that hang."""
    sk = Skeleton()
    top = h * 0.05
    lean = rng.uniform(-0.04, 0.04)
    lead = stem(sk, w / 2 - 0.5, h - 2, top, rng, wobble=0.1, lean=lean)
    low = h * 0.62
    pts = room(rng, w / 2 + lean * h * 0.5, h * 0.32, w * 0.38, h * 0.3, clumps=(5, 7), per=(14, 24), under=0.9,
               upper_only=0.3)
    colonise(sk, pts, rng, set_angle=35, set_k=0.5, wobble=0.15, reach=11,
             start=[i for i in lead if sk.pos[i][1] < low])
    return sk, dict(base=1.9, leaves="clumps", clump=(1.6, 2.6), lobes=(7, 10), fine=0.8, low=0.62,
                    hang=(4, 10, 0.8))


def grow_willow(rng, w, h):
    """Decurrent, often with more than one stem from the base; a broad dome whose branches,
    with age, bend over and hang in long curtains."""
    sk = Skeleton()
    stems = rng.choice([1, 2, 2, 3, 3])
    tops = []
    for k in range(stems):
        off = k - (stems - 1) / 2                          # the outer stems splay out from one foot
        x = w / 2 - 0.5 + off * 2.2
        ids = stem(sk, x, h - 2, h * rng.uniform(0.5, 0.6), rng, wobble=0.12,
                   lean=off * rng.uniform(0.16, 0.24))
        tops += ids[len(ids) // 2:]
    pts = room(rng, w / 2, h * 0.36, w * 0.48, h * 0.3, under=0.6, shell=0.5)
    colonise(sk, pts, rng, set_angle=50, set_k=0.3, wobble=0.2, start=tops)
    return sk, dict(base=2.6, leaves="clumps", clump=(3.0, 4.2), lobes=(6, 8), fine=1.1, low=0.7,
                    hang=(8, 18, 0.8))


def grow_windbent(rng, w, h):
    """A flag tree: the wind from the west kills the branches on that side; what is left leans
    and points east, low over the ground."""
    sk = Skeleton()
    base = stem(sk, w * 0.32, h - 2, h * 0.5, rng, lean=0.28, wobble=0.1)
    xs = sk.pos[base[-1]][0]
    pts = room(rng, w * 0.6, h * 0.36, w * 0.4, h * 0.3, under=0.6, keep=lambda x, y: x > xs - 3)
    colonise(sk, pts, rng, set_angle=75, set_k=0.4, pull=(0.35, 0.05), wobble=0.25, start=base[len(base) // 2:])
    return sk, dict(base=2.4, leaves="clumps", clump=(3.0, 4.4), lobes=(4, 6), fine=1.1, low=0.66)


def grow_pine(rng, w, h, old=False):
    """Monopodial, in whorls: the leader puts out a ring of near-level branches each year, and
    the lower ones are older and longer - a cone. With age the lower whorls die and fall (a long
    bare stem), the leader slows, and the last few limbs spread into a flat top. The needles sit
    in flat pads at the branch ends."""
    sk = Skeleton()
    top = h * (0.12 if old else 0.03)
    lead = stem(sk, w / 2 - 0.5, h - 2, top, rng, wobble=0.12 if old else 0.04)
    ys = np.array([sk.pos[i][1] for i in lead])
    gap = rng.uniform(4.2, 5.2)
    whorl_y, y = [], top + 2.5
    while y < h * (0.42 if old else 0.86):
        whorl_y.append(y)
        y += gap * (1.0 + 0.04 * len(whorl_y))
    for wy in whorl_y:
        node = lead[int(np.argmin(np.abs(ys - wy)))]
        depth = (wy - top) / (h - top)
        if old:
            length = w * rng.uniform(0.32, 0.5) * (0.7 + 0.6 * min(depth / 0.3, 1))
        else:
            length = 2 + (w * 0.5 - 2) * depth ** 0.9
        for side in (-1, 1):
            if rng.random() < (0.85 if old else 0.95):
                b = limb(sk, node, (side * 1.0, -0.7 if old else -0.25), length * rng.uniform(0.75, 1.05), rng,
                         bend=side * rng.uniform(1.5, 4), droop=0.15 + depth * 0.25, wobble=0.15 if old else 0.06)
                if length > 7:                             # a side shoot or two from the longer branches
                    for j in range(1, len(b) - 2, rng.randint(3, 5)):
                        limb(sk, b[j], (side * 0.6, rng.choice([-0.6, 0.4])), length * 0.3, rng)
    # (an old pine: needles only at the branch ends, so its limbs and the orange bark of its upper
    # stem - Scots pine's mark - show between the pads and all down the top of the bare trunk)
    return sk, dict(base=2.6 if old else 2.1, leaves="pads", clump=(2.4, 3.4) if old else (2.6, 3.6),
                    lobes=(len(whorl_y) + 3, len(whorl_y) + 6), fine=0.62 if old else 1.0, low=0.5 if old else 0.92,
                    up_from=0.7 if old else None)


SPECIES = {
    "oak": (grow_oak, 56, 62),
    "alder": (grow_alder, 34, 60),
    "birch": (grow_birch, 34, 62),
    "willow": (grow_willow, 60, 56),
    "windbent": (grow_windbent, 48, 40),
    "pine": (grow_pine, 32, 62),
    "pine_old": (lambda rng, w, h: grow_pine(rng, w, h, old=True), 48, 64),    # (sprites are at most 64 px)
}


# ---------------------------------------------------------------- drawing

def _bark(idx, sk, r, w, h, up_from=None):
    """The limbs as thick lines, each lit on its left side."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    best = np.full((h, w), np.inf)
    side = np.zeros((h, w))
    for i, p in enumerate(sk.parent):
        if p < 0:
            continue
        (ax, ay), (bx, by) = sk.pos[p], sk.pos[i]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy + 1e-9
        t = np.clip(((xx - ax) * dx + (yy - ay) * dy) / L2, 0, 1)
        px, py = ax + t * dx, ay + t * dy
        rad = np.maximum(r[p] + (r[i] - r[p]) * t, 0.5)
        rel = np.hypot(xx - px, yy - py) - rad
        m = (rel < 0) & (rel < best)
        best[m] = rel[m]
        nx, ny = -dy, dx                                   # the normal facing left (and up) is lit
        if nx > 0 or (nx == 0 and ny > 0):
            nx, ny = -nx, -ny
        nl = math.hypot(nx, ny) + 1e-9
        u = ((xx - px) * nx + (yy - py) * ny) / nl / rad
        side[m] = u[m]
    m = np.isfinite(best)
    thin = m & (best > -0.9)                               # a pixel wide: no room for light
    tone = np.where(side > 0.45, 3, np.where(side > -0.1, 2, np.where(side > -0.6, 1, 0)))
    tone = np.where(thin, 1, tone)
    if up_from is not None:
        # the orange bark flakes in above the grey, ragged over a few rows - not a painted line
        hi = yy < h * up_from + (_hash(np.floor(xx), np.floor(yy / 2), 19) - 0.5) * 8
        idx[m & hi] = np.array(U)[tone[m & hi]]
        idx[m & ~hi] = np.array(B)[tone[m & ~hi]]
    else:
        idx[m] = np.array(B)[tone[m]]
    return m & ~thin


def _kmeans(X, k, rng, it=12):
    k = max(1, min(k, len(X)))
    C = X[rng.sample(range(len(X)), k)].copy()
    for _ in range(it):
        lab = np.hypot(X[:, None, 0] - C[None, :, 0], X[:, None, 1] - C[None, :, 1]).argmin(axis=1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(axis=0)
    return lab, C


def _bark_marks(idx, thick, kind, seed, h):
    """Bark you can tell apart: oak, pine and willow in vertical furrows; birch white with dark
    dashes across and a dark, cracked foot; alder smooth and grey."""
    yy, xx = np.mgrid[0:idx.shape[0], 0:idx.shape[1]]
    if kind in ("oak", "pine", "pine_old", "willow", "windbent"):
        furrow = (_hash(xx, np.floor(yy / 3), seed + 7) < 0.22) & np.isin(idx, B[1:]) & thick
        idx[furrow] = np.where(idx[furrow] == B[3], B[2], B[1])
    if kind == "birch":
        dash = (_hash(np.floor(xx / 2), yy, seed + 3) < 0.13) & thick & (yy % 2 == 0)
        idx[dash] = B[0]
        foot = thick & (yy > h * 0.85) & (_hash(xx, np.floor(yy / 2), seed + 5) < 0.55)
        idx[foot] = B[0]


def tree(P, kind="oak", seed=0, leaves=True, season="summer", size=None):
    """One tree. size: its age as a share of a full-grown one (0.6-1); by default the seed picks
    it, so trees of one kind stand at different heights."""
    grow, W, H = SPECIES[kind]
    rng = random.Random(f"grow-{kind}:{seed}")
    f = size if size is not None else 0.7 + 0.3 * rng.random()
    w, h = max(12, round(W * (0.55 + 0.45 * f))), max(16, round(H * f))   # younger: narrower, but less so than lower
    sk, spec = grow(rng, w, h)
    spec = {**spec, "base": spec["base"] * f ** 0.6, "clump": tuple(c * max(f, 0.75) for c in spec["clump"])}
    leaf_kind = {"pine_old": "pine", "windbent": "oak"}.get(kind, kind)
    names = LEAF[leaf_kind] + BARK[leaf_kind] + BARK_UP.get(leaf_kind, BARK[leaf_kind])
    if season == "autumn":
        names[:5] = {"oak": ["w0", "o0", "y1", "y1", "y2"], "birch": ["w1", "o0", "y1", "y2", "y2"],
                     "alder": ["b0", "g0", "g1", "w2", "w3"], "willow": ["w0", "w1", "w3", "y1", "y2"]}.get(
            leaf_kind, names[:5])
    r = sk.radii(spec["base"])
    idx = np.zeros((h, w), dtype=np.int16)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    thick = _bark(idx, sk, r, w, h, spec.get("up_from"))
    flare = np.zeros((h, w), dtype=bool)
    for i, p in enumerate(sk.parent):
        if p < 0:
            rx0 = sk.pos[i][0]
            flare |= (np.abs(xx - rx0) < r[i] + (yy - (h - 4)) * 1.1) & (yy > h - 4) & (yy < h - 1)
    idx[flare & (idx == T)] = B[1]
    _bark_marks(idx, thick, kind, seed, h)
    no_leaf = np.zeros_like(thick)
    fine = [i for i in range(len(sk.pos)) if r[i] < spec["fine"] and sk.pos[i][1] < h * spec["low"]]
    tips = np.array([sk.pos[i] for i in fine]).reshape(-1, 2)
    if not leaves or not len(tips):
        return _finish(idx, thick | flare, no_leaf, names, P)
    lab, C = _kmeans(tips, rng.randint(*spec["lobes"]), rng)
    # a lobe of only a twig or two would be lit as a little ball of its own and stand out as a pale
    # bubble: it joins the nearest proper lobe and is lit as part of it
    counts = np.bincount(lab, minlength=len(C))
    big = np.nonzero(counts >= 3)[0]
    if len(big):
        for j in np.nonzero((counts > 0) & (counts < 3))[0]:
            to = big[np.argmin(np.hypot(C[big, 0] - C[j, 0], C[big, 1] - C[j, 1]))]
            lab[lab == j] = to
        for j in big:
            C[j] = tips[lab == j].mean(axis=0)
    # a lobe much bigger than the rest has nothing in front of it to shade it and reads as one
    # smooth pale mass: part it in two, one in front of the other, like the others
    counts = np.bincount(lab, minlength=len(C))
    typical = np.median(counts[counts > 0])
    C = list(C)
    for j in np.nonzero(counts > 1.5 * typical)[0]:
        sub, SC = _kmeans(tips[lab == j], 2, rng)
        where = np.nonzero(lab == j)[0]
        lab[where[sub == 1]] = len(C)
        C[j] = SC[0]
        C.append(SC[1])
    C = np.array(C)
    crown_r = max(np.ptp(tips[:, 0]), np.ptp(tips[:, 1])) / 2 + spec["clump"][1]
    owner = np.full((h, w), -1, dtype=np.int16)
    light1 = np.zeros((h, w))
    light3 = np.zeros((h, w))
    thin_leaf = np.zeros((h, w), dtype=bool)
    pads = spec["leaves"] == "pads"
    lo, hi = spec["clump"]
    lobe_at = {}
    for rank, j in enumerate(np.argsort(C[:, 1])):        # higher lobes first, lower in front
        TT = tips[lab == j]
        if not len(TT):
            continue
        mask = np.zeros((h, w), dtype=bool)
        for tx, ty in TT:
            cr = lo + _hash(tx, ty, seed) * (hi - lo)
            if pads:                                       # flat needle pads, domed a little on top
                mask |= np.hypot((xx - tx) / 1.25, (yy - ty + 0.6) * 1.9) < cr * 0.85
            else:
                disc = np.hypot(xx - tx, (yy - ty) * 1.1) < cr
                mask |= disc
                # each twig's own spray, a small ball inside the lobe: the lobe reads as made of
                # sprays everywhere, not as one smooth bubble
                light3[disc] = (-((xx - tx) * LX + (yy - ty) * LY) / cr)[disc]
        lx, ly = TT.mean(axis=0)
        lr = max(3.5, crown_r * 0.33, np.hypot(TT[:, 0] - lx, TT[:, 1] - ly).max() + lo)   # never a tiny ball
        sy = 2.2 if pads else 1.0                          # a pad is lit on its top, dark beneath
        lobe_at[rank] = (lx, ly, lr, sy)
        owner[mask] = rank
        light1[mask] = (-((xx - lx) * LX + (yy - ly) * sy * LY) / lr)[mask]
        if spec.get("hang"):                               # hanging twigs: strands from the tips
            a, b, share = spec["hang"]
            for tx, ty in TT:
                if _hash(tx, ty, seed + 11) > share:
                    continue
                n = int(a + _hash(ty, tx, seed) * (b - a))
                out = 0.25 if tx > lx else -0.25
                for s in range(1, n):
                    x, y = int(round(tx + out * math.sqrt(s))), int(round(ty + s))
                    if 0 <= x < w and 0 <= y < h - 3 and owner[y, x] < 0:
                        owner[y, x] = rank
                        light1[y, x] = -((x - lx) * LX + (y - ly) * LY) / lr - s * 0.05
                        thin_leaf[y, x] = True
    if not pads:
        _join_lobes(owner, light1, lobe_at, lo, xx, yy)
    crown = owner >= 0
    cys, cxs = np.nonzero(crown.any(axis=1))[0], np.nonzero(crown.any(axis=0))[0]
    ccx, ccy = xx[crown].mean(), (cys.min() + cys.max()) / 2
    rxx, ry = max((cxs.max() - cxs.min()) / 2, 1), max((cys.max() - cys.min()) / 2, 1)
    light2 = -((xx - ccx) * LX + (yy - ccy) * LY) / max(rxx, ry)
    under = np.clip((yy - ccy) / (ry * 0.95), 0, 1) ** 1.3 * (0.4 if pads else 1.0)
    shade = np.zeros((h, w), dtype=bool)
    for ox, oy in ((1, 2), (2, 2), (0, 2), (2, 1)):
        ahead = np.full((h, w), -1, dtype=np.int16)
        ahead[:h - oy, :w - ox] = owner[oy:, ox:]
        shade |= crown & (ahead > owner)
    # and the light from above: a lobe tucked under another is in its shadow along its top
    for k in (1, 2):
        above = np.full((h, w), -1, dtype=np.int16)
        above[k:, :] = owner[:-k, :]
        shade |= crown & (above >= 0) & (above != owner) & (above < owner)
    cell = _hash(np.floor((xx + (np.floor(yy / 2) % 2) * 1.5) / 3), np.floor(yy / 2), seed) - 0.5
    v = 0.5 * light1 + 0.4 * light2 + 0.3 * light3 - 0.8 * shade - 0.6 * under + 0.36 * cell
    tone = np.digitize(v, [-0.8, -0.32, 0.12, 0.55])
    if not pads:
        _hold_up(idx, owner, lobe_at, sk, r, w, h)
    idx = np.where(crown, np.array(L)[tone], idx)
    below = _shift(crown, 1, 0) | _shift(crown, 2, 0)
    idx[below & ~crown & np.isin(idx, B[2:] + U[2:])] = B[1]
    return _finish(idx, thick | flare, thin_leaf, names, P)


def _join_lobes(owner, light1, lobe_at, r, xx, yy):
    """A lobe that touches no other floats beside the crown like a pale ball, its branch hidden
    behind the leaves: leaf it along the way to the nearest lobe, as the branch would be."""
    h, w = owner.shape
    for rank, (lx, ly, lr, sy) in lobe_at.items():
        m = owner == rank
        if not m.any():
            continue
        crown = owner >= 0
        near = m.copy()
        for oy, ox in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            near |= _shift(m, oy, ox)
        others = crown & ~m
        if not others.any() or (near & others).any():
            continue
        ay, ax = np.nonzero(m)
        by, bx = np.nonzero(others)
        d = np.hypot(ax[:, None] - bx[None, :], ay[:, None] - by[None, :])
        i, j = np.unravel_index(d.argmin(), d.shape)
        n = int(d[i, j]) + 2
        for t in np.linspace(0, 1, n):
            cx, cy = ax[i] + (bx[j] - ax[i]) * t, ay[i] + (by[j] - ay[i]) * t
            disc = (np.hypot(xx - cx - 0.5, (yy - cy - 0.5) * 1.1) < r * 0.8) & (owner < 0)
            owner[disc] = rank
            light1[disc] = (-((xx - lx) * LX + (yy - ly) * sy * LY) / lr)[disc]


def _hold_up(idx, owner, lobe_at, sk, r, w, h):
    """A lobe with air under it needs a branch you can see holding it up. Space colonisation may
    have fed it from above, behind other leaves; draw a limb from under its middle down to the
    nearest big limb or the trunk. It goes in under the leaves, so it shows only in the gap."""
    big = [i for i in range(len(sk.pos)) if r[i] >= 1.3]
    if not big:
        return
    B_ = np.array([sk.pos[i] for i in big])
    crown = owner >= 0
    for rank, (lx, ly, lr, sy) in lobe_at.items():
        m = owner == rank
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        cx = int(round(np.median(xs)))
        col = ys[xs == cx]
        if not len(col):
            continue
        y0 = col.max()                                     # the lobe's underside, mid-way across
        below = crown[y0 + 1:min(y0 + 4, h), cx]
        if below.any():
            continue                                       # it rests on other leaves
        cand = B_[B_[:, 1] > y0 + 2]
        if not len(cand):
            continue
        tx, ty = cand[np.argmin(np.hypot(cand[:, 0] - cx, cand[:, 1] - y0))]
        bone = Skeleton()
        a = bone.add(tx, ty, -1)
        bone.add((tx + cx) / 2, (ty + y0) / 2 + 0.5, a)
        bone.add(cx, y0 - 1, 1)
        tmp = np.zeros_like(idx)
        _bark(tmp, bone, np.array([1.7, 1.3, 1.0]), w, h)
        idx[(tmp != T) & (idx == T)] = tmp[(tmp != T) & (idx == T)]


def _finish(idx, thick, thin_leaf, names, P):
    """The outline: dark green round the leaves (a step lighter where the light falls), ink round
    the trunk and big limbs - none round a twig or a hanging strand, which would double it."""
    idx = _clean(idx, keep=B + U)
    filled = idx != T
    edge = np.zeros_like(filled)
    for oy, ox in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        edge |= _shift(filled, oy, ox)
    edge &= ~filled
    lit_side = _shift(np.isin(idx, L[3:]), -1, -1)
    body = np.isin(idx, L) & ~thin_leaf
    leaf_near = np.zeros_like(filled)
    thick_near = np.zeros_like(filled)
    for oy, ox in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        leaf_near |= _shift(body, oy, ox)
        thick_near |= _shift(thick, oy, ox)
    idx = np.where(edge & leaf_near, np.where(lit_side, L[1], L[0]), idx)
    idx = np.where(edge & ~leaf_near & thick_near, B[0], idx)
    return _image(idx, names, P)
