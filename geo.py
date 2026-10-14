"""Makes maps the way land is made, instead of by drawing blobs.

The laws that shape land work at the scale of a whole valley - water gathers into rivers,
rock lies in bands, rivers sort what they carry - and our maps are a few dozen steps across.
So there are two levels:

**The valley** (`valley`) is worked out whole, on a coarse grid where a cell is some tens of
paces. It is never played on; it is the reason for everything.

Its shape is not drawn but grown (`_evolved`): the ground is pushed up, and rivers cut it down,
for a few hundred thousand years, until the two balance. That is a landscape-evolution model:
uplift against stream-power incision (solved implicitly down the drainage, after Braun &
Willett 2013), hillslope creep, and the valley floors filled where the rivers slow. Real
valleys measured by survey (`dem/`, see dem/sources.json and docs/引用数据.md) are not copied:
they are what it is calibrated against - how much of the land is steep, how much flat, how
the height is shared out. (`source=<name>` still puts a real valley in its place, to compare.)
On that land, in the order it happens:

1. **Rock and uplift.** A basin open to the sea on the south, high ground on the other three
   sides. A fault crosses it and the landward block has been thrown up: the scarp. The rock
   lies in bands along the fault's strike, hard and soft by turns. The land rises; rivers cut
   into it as it rises, most where most water gathers; slopes creep between them. The sea
   has since risen a little and drowned the mouths of the valleys.
2. **Rain and run-off.** Rain falls, more on high ground and on slopes facing the wind off
   the sea. Water runs down the steepest way; hollows fill until they spill; where enough has
   gathered there is a stream, then a river.
3. **Erosion.** Running water cuts down by how much of it there is and how steeply it falls,
   and much faster in soft rock than hard. Valleys follow the soft bands; hard bands stand as
   ledges; a river steepens where it crosses one. Slopes creep, and the too-steep ones slump.
4. **Deposition.** A river carries what it has cut and drops it where it slows, coarse first:
   stones in the steep reaches, sand below, mud on the flats and at the mouth. The sea lays
   sand in the bays between headlands of hard rock, and mud where a river mouth shelters it.
5. **Climate.** How wet it is decides how many streams, how deep the soil, what grows: woods
   where it is damp, sheltered and the soil holds; meadow where it is drier or the sea wind
   scours; bare rock where it is too steep for soil.

**A place** (`place`) is a window on the valley blown up to walking scale: one cell a step.
Its heights, rock, damp and sediment are the valley's, read finer; its river is the valley's
river, drawn at its real width. Nothing in it is made up locally - a patch of gravel is there
because the river is steep there, and it is steep there because of the hard band upstream.

Out comes a Map with a height for every cell - `relief`, 0 for the sea and up to LEVELS * FINE
at the valley's top, counted across the whole valley - and an
account of why the place is the way it is.
"""
from __future__ import annotations

import functools
import heapq
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import mapkit
from mapkit import MapError

# How wet. rain: run-off; streams: share of the land that is channel; wood: how readily trees
# take hold (lower = more); soil: how fast rock weathers to soil; marsh: how much low wet ground.
CLIMATES = {
    "oceanic": {"name": "温带海洋性", "rain": 1.0, "streams": 0.045, "wood": 0.5, "soil": 1.0, "marsh": 1.0, "treeline": 0.55,
                "about": "凉、常年湿润、多风。溪流密，土层厚，背风处成林，迎海一面被风压成草甸。"},
    "dry": {"name": "夏干（地中海式）", "rain": 0.55, "streams": 0.022, "wood": 0.63, "soil": 0.6, "marsh": 0.3, "treeline": 0.7,
            "about": "夏季干旱。溪流少而急，裸岩和卵石滩多，林子只在沟谷和背阴处。"},
    "cold": {"name": "亚寒带", "rain": 0.8, "streams": 0.036, "wood": 0.74, "soil": 0.45, "marsh": 1.6, "treeline": 0.3,
             "about": "冷，生长季短。土层薄，树很少，低处多沼泽，岩石裸露。"},
}
NEAR8 = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))
VALLEY = (128, 96)   # cells of the valley grid
ZOOM = 5             # map cells to one valley cell
LEVELS = 12          # heights are counted in this many steps from the sea to the valley's top
FINE = 20            # ...and stored FINE times finer than that, so a river's fall shows within one step
DEM = Path(__file__).resolve().parent / "dem"


def sources() -> dict:
    """The real valleys there are to choose from: name -> {name, climate, about, cell_m, ...}."""
    path = DEM / "sources.json"
    if not path.exists():
        return {}
    # only those whose heights are here (dem/fetch.py downloads them)
    return {k: v for k, v in json.loads(path.read_text(encoding="utf-8"))["valleys"].items() if (DEM / f"{k}.npy").exists()}


PLACES = {"mouth": "河口", "river": "河的中游", "gorge": "河穿过硬岩的陡段", "scarp": "断层陡坎", "bay": "海湾沙滩",
          "headland": "岬角", "upland": "高处"}


def _smooth_noise(rng, w, h, cell):
    """Value noise 0..1 with features about `cell` wide."""
    gw, gh = int(w / cell) + 2, int(h / cell) + 2
    lat = rng.random((gh, gw)).astype(np.float32)
    x, y = np.arange(w) / cell, np.arange(h) / cell
    ix, iy = x.astype(int), y.astype(int)
    tx, ty = x - ix, y - iy
    tx, ty = (tx * tx * (3 - 2 * tx))[None, :], (ty * ty * (3 - 2 * ty))[:, None]
    top = lat[np.ix_(iy, ix)] * (1 - tx) + lat[np.ix_(iy, ix + 1)] * tx
    bot = lat[np.ix_(iy + 1, ix)] * (1 - tx) + lat[np.ix_(iy + 1, ix + 1)] * tx
    return top * (1 - ty) + bot * ty


def _vnoise(a, b, salt):
    """Smooth value noise 0..1 read at float coordinates (a, b), lattice 1 apart."""
    ia, ib = np.floor(a), np.floor(b)
    fa, fb = a - ia, b - ib
    fa, fb = fa * fa * (3 - 2 * fa), fb * fb * (3 - 2 * fb)

    def h(i, j):
        v = np.sin(i * 127.1 + j * 311.7 + salt * 74.7) * 43758.5453
        return v - np.floor(v)

    top = h(ia, ib) * (1 - fa) + h(ia + 1, ib) * fa
    bot = h(ia, ib + 1) * (1 - fa) + h(ia + 1, ib + 1) * fa
    return top * (1 - fb) + bot * fb


def _blur(a, radius):
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return a.astype(np.float32)
    img = Image.fromarray(np.rint((a - lo) / (hi - lo) * 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255 * (hi - lo) + lo


def _spread(mask, steps):
    """Distance (in steps, up to `steps`) of every cell from `mask`; steps+1 beyond that."""
    d = np.full(mask.shape, steps + 1, dtype=np.int16)
    d[mask] = 0
    reach = mask.copy()
    for i in range(1, steps + 1):
        grown = reach.copy()
        grown[1:, :] |= reach[:-1, :]
        grown[:-1, :] |= reach[1:, :]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        d[grown & ~reach] = i
        reach = grown
    return d


def _drainage(z, sea):
    """Fill hollows until they spill, then send every cell's water to its lowest neighbour.
    Returns (filled heights, receiver index per cell or -1, cells highest first, slope)."""
    h, w = z.shape
    done = np.zeros((h, w), dtype=bool)
    done[sea] = True
    done[-1, :] = True
    ys, xs = np.nonzero(done)
    zl = z.astype(np.float64).tolist()
    heap = [(zl[y][x], y, x) for y, x in zip(ys.tolist(), xs.tolist())]
    heapq.heapify(heap)
    dl = done.tolist()
    while heap:
        c, y, x = heapq.heappop(heap)
        for dx, dy, _ in NEAR8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not dl[ny][nx]:
                dl[ny][nx] = True
                if zl[ny][nx] <= c:
                    zl[ny][nx] = c + 1e-5
                heapq.heappush(heap, (zl[ny][nx], ny, nx))
    zf = np.array(zl)
    best = np.zeros((h, w))
    recv = np.full((h, w), -1, dtype=np.int64)
    pad = np.pad(zf, 1, constant_values=np.inf)
    for dx, dy, dist in NEAR8:
        drop = (zf - pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]) / dist
        better = drop > best
        best[better] = drop[better]
        ys, xs = np.nonzero(better)
        recv[ys, xs] = (ys + dy) * w + (xs + dx)
    recv[sea] = -1
    return zf, recv, np.argsort(-zf, axis=None, kind="stable"), best


# ---------------------------------------------------------------- the valley

def _work(z, hard, rain):
    """One round of water on the land: where it cuts, where it drops what it carries (and how
    fast it was going when it did), and what slumps from slopes too steep to stand.
    Returns the change in height from water, the change from slumping, and the fields."""
    H, W = z.shape
    sea = z < 0
    _, recv, order, slope = _drainage(z, sea)
    rl, sl = recv.ravel().tolist(), slope.ravel().tolist()
    water, load = rain.ravel().tolist(), [0.0] * (H * W)
    zl, hl, dz = z.ravel().tolist(), hard.ravel().tolist(), [0.0] * (H * W)
    for i in order.tolist():
        if zl[i] < 0:
            continue
        a, s, r = water[i], sl[i], rl[i]
        carry = 0.055 * (a ** 0.55) * s                 # what this much water at this slope can move
        if load[i] > carry:
            drop = (load[i] - carry) * 0.3
            dz[i] += drop
            load[i] -= drop
        else:
            take = min((carry - load[i]) * 0.6 / hl[i], 0.02 / hl[i], zl[i] * 0.5)
            dz[i] -= take
            load[i] += take
        if r >= 0:
            water[r] += a
            if zl[r] < 0:
                dz[r] += load[i] * 0.7                  # the sea: it comes to rest at the mouth
            else:
                load[r] += load[i]
    dz = np.array(dz).reshape(H, W)
    slump, scree = np.zeros((H, W)), np.zeros((H, W))
    hold = 0.045 + 0.1 * hard                           # hard rock holds a steeper face
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        lower = np.roll(z, (-dy, -dx), axis=(0, 1))
        over = np.clip(z - lower - hold, 0, None) * 0.25
        over[z < 0] = 0
        if dx == 1:
            over[:, -1] = 0
        if dx == -1:
            over[:, 0] = 0
        if dy == 1:
            over[-1, :] = 0
        if dy == -1:
            over[0, :] = 0
        fell = np.roll(over, (dy, dx), axis=(0, 1))
        slump += fell - over
        scree += fell * np.roll(hard, (dy, dx), axis=(0, 1))
    return dz, slump, slope, scree


def _drown(metres, share):
    """The sea let in from the south edge, high enough to cover `share` of the map: what it
    covers, and the level it stands at."""
    def flooded(level):
        low = metres <= level
        sea = np.zeros_like(low)
        sea[-1, :] = low[-1, :]
        while True:
            grown = sea.copy()
            grown[:-1, :] |= sea[1:, :]
            grown[1:, :] |= sea[:-1, :]
            grown[:, 1:] |= sea[:, :-1]
            grown[:, :-1] |= sea[:, 1:]
            grown &= low
            if (grown == sea).all():
                return sea
            sea = grown

    lo = float(metres[-1].min())
    hi = lo + float(np.quantile(metres, 0.5) - lo)
    for _ in range(12):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if flooded(mid).mean() < share else (lo, mid)
    return flooded(hi), hi


def _evolved(rng, C, steps=100):
    """Steps 1 and 3 by letting them happen, on a basin of no place in particular."""
    W, H = VALLEY
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    u, v = xx / W, yy / H
    # uplift: a basin rimmed on three sides, open to the sea on the south; a fault throws the
    # landward block up faster; noise, so the water finds its own ways
    strike = math.radians(float(rng.uniform(-14, 14)))
    across = (-(xx - W / 2) * math.sin(strike) + (yy - H / 2) * math.cos(strike)) / H + 0.5
    warp = _smooth_noise(rng, W, H, H / 2.2) - 0.5
    mid = 0.5 + 0.12 * math.sin(float(rng.uniform(0, 6))) + 0.08 * (v - 0.5)
    rim = np.clip(1 - v, 0, 1) ** 0.8 * 0.7 + 0.9 * np.abs(u - mid) ** 1.5 * 1.6
    up = 0.25 + rim + 0.35 * (across + 0.08 * warp < float(rng.uniform(0.38, 0.5)))
    up *= 1 + 0.35 * (_smooth_noise(rng, W, H, H / 4) - 0.5)
    beds = np.sin(2 * math.pi * (across / 0.3 + 0.5 * warp + float(rng.uniform(0, 1))))
    hard = np.where(_blur(beds, 0.8) > 0.2, 1.0, 0.3).astype(np.float32)
    base = v > 0.86 + 0.05 * (_smooth_noise(rng, W, H, W / 5) - 0.5)          # the sea, then: base level
    h = _grow(up, hard, base, rng, C, steps)
    # since then the sea has risen a little and drowned the valley mouths
    relief_m, cell_m = 800.0, 62.0
    metres = h / max(float(np.quantile(h[~base], 0.995)), 1e-9) * relief_m
    sea, level = _drown(metres, 0.14)
    relief = float(np.quantile(metres[~sea], 0.995) - level)
    z = np.where(sea, -0.02 - 0.012 * _spread(~sea, 8), np.maximum(metres - level, 0.3) / relief)
    return (z, hard, z.copy(), relief, cell_m,
            f"生成的盆地：抬升与河流下切相抗约 80 千年（模型时间），岩层与断层走向偏 {math.degrees(strike):+.0f}°，"
            "断层靠山一侧抬得快；谷底按河的大小淤平，之后海面上升淹了河口。")


def _grow(up, hard, base, rng, C, steps=100, fill=2.0):
    """Uplift `up` against river incision and creep, the sea (`base`) held at 0; hard rock
    wears slower. Returns the heights.

    Every step: hollows are filled until they spill (priority-flood from the sea), every cell
    drains down its steepest way, the area draining through each cell is summed, and the
    stream-power law dh/dt = U - K A^m S is solved implicitly from the sea upwards; then the
    slopes creep (diffusion). Last, the floors fill: what a river brings down settles where it
    slows, so next to a big river the land lies nearly level with it."""
    H, W = up.shape
    m, K, Kratio, D = 0.5, 3.0 * (0.7 + 0.3 * C["rain"]), 0.6, 0.4
    Kl = (np.where(hard > 0.5, K * Kratio, K) * 1e-4).ravel().tolist()        # hard beds wear slower
    bl = base.ravel()
    h = 0.02 * rng.random((H, W))
    h[base] = 0.0
    dt = 80000.0 / steps                                                       # the same span, however finely cut
    Ul = (up * 1e-3 * dt).ravel().tolist()                                      # uplift over one step
    idx = np.arange(H * W).reshape(H, W)
    seeds = np.flatnonzero(bl).tolist()
    def downhill(zz, eps):
        """Each cell's steepest way down over `zz`; a whisker of chance (less than the flood's
        step, so no flat becomes a pit) keeps channels off the grid lines."""
        jit = zz + eps * 0.4 * rng.random((H, W))
        pad = np.pad(jit, 1, constant_values=np.inf)
        best, recv, dist = np.zeros((H, W)), np.full((H, W), -1, dtype=np.int64), np.ones((H, W))
        for dx, dy, dd in NEAR8:
            drop = (jit - pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]) / dd
            better = drop > best
            best[better], recv[better], dist[better] = drop[better], idx[better] + dy * W + dx, dd
        rl = recv.ravel()
        rl[bl] = -1
        return rl, dist

    for _ in range(steps):
        hl = h.ravel().tolist()
        eps = 1e-5 * (1 + float(h.max()))
        rl, dist = downhill(h, eps)
        lost = np.flatnonzero((rl < 0) & ~bl)
        if len(lost):
            # hollows: fill them until they spill (priority-flood from the sea), and drain over that
            zf, done, parent = list(hl), bl.tolist(), [-1] * (H * W)
            heap = [(hl[i], i) for i in seeds]
            heapq.heapify(heap)
            while heap:
                c, i = heapq.heappop(heap)
                y, x = divmod(i, W)
                for dx, dy, _ in NEAR8:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        j = ny * W + nx
                        if not done[j]:
                            done[j] = True
                            zf[j] = max(zf[j], c + eps)
                            parent[j] = i
                            heapq.heappush(heap, (zf[j], j))
            rl, dist = downhill(np.array(zf).reshape(H, W), eps)
            lost = np.flatnonzero((rl < 0) & ~bl)
            rl[lost] = np.array(parent)[lost]
        rl, dl = rl.tolist(), dist.ravel().tolist()
        donors = [[] for _ in range(H * W)]
        for i, r in enumerate(rl):
            if r >= 0:
                donors[r].append(i)
        stack = list(seeds)
        k = 0
        while k < len(stack):
            stack.extend(donors[stack[k]])
            k += 1
        A = [1.0] * (H * W)
        for i in reversed(stack):
            if rl[i] >= 0:
                A[rl[i]] += A[i]
        for i in stack:
            r = rl[i]
            if r >= 0:
                F = Kl[i] * dt * A[i] ** m / dl[i]
                hl[i] = (hl[i] + Ul[i] + F * hl[r]) / (1 + F)
        h = np.array(hl).reshape(H, W)
        creep = int(math.ceil(D * 1e-3 * dt / 0.2))                           # sub-steps, so creep stays stable
        for _ in range(creep):
            hp = np.pad(h, 1, mode="edge")
            h = h + D * 1e-3 * dt / creep * (hp[:-2, 1:-1] + hp[2:, 1:-1] + hp[1:-1, :-2] + hp[1:-1, 2:] - 4 * h) * (~base)
        h[base] = 0.0
    # the floors fill: height above the channel the water runs to decides it; the bigger the
    # river, the wider and deeper the fill, easing out up the valley sides
    A = np.array(A).reshape(H, W)
    Ac, Amax = float(np.quantile(A, 0.985)), float(A.max())
    hl, Al = h.ravel().tolist(), A.ravel().tolist()
    chan_h, chan_a = [0.0] * (H * W), [0.0] * (H * W)
    for i in stack:
        r = rl[i]
        if Al[i] > Ac or r < 0:
            chan_h[i], chan_a[i] = hl[i], Al[i]
        else:
            chan_h[i], chan_a[i] = chan_h[r], chan_a[r]
    ch, ca = np.array(chan_h).reshape(H, W), np.array(chan_a).reshape(H, W)
    above = h - ch
    reach = fill * 0.08 * float(h.max()) * np.clip(np.log(ca / Ac + 1) / np.log(Amax / Ac + 1), 0, 1)
    return h - np.clip(1 - above / np.maximum(reach, 1e-9), 0, 1) ** 0.6 * (ca > Ac) * above * 0.88


DESIGNS = Path(__file__).resolve().parent / "designs"


def designs() -> dict:
    """The designed valleys there are: name -> the design (designs/<name>.json)."""
    return {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sorted(DESIGNS.glob("*.json"))} if DESIGNS.exists() else {}


def _polyline_v(points, u):
    """The v of a polyline [[u, v], ...] (sorted by u) at each u."""
    us, vs = zip(*points)
    return np.interp(u, us, vs)


def _to_line(points, u, v):
    """Distance (in map widths, v scaled like u) from each (u, v) to a polyline."""
    best = np.full(u.shape, np.inf)
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        dx, dy = bx - ax, by - ay
        t = np.clip(((u - ax) * dx + (v - ay) * dy) / max(dx * dx + dy * dy, 1e-12), 0, 1)
        best = np.minimum(best, np.hypot(u - ax - t * dx, v - ay - t * dy))
    return best


def _inside(poly, u, v):
    """Whether each (u, v) lies inside the closed polygon [[u, v], ...] (even-odd rule)."""
    out = np.zeros(u.shape, dtype=bool)
    for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1]):
        cross = (ay > v) != (by > v)
        at = ax + (v - ay) * (bx - ax) / ((by - ay) or 1e-12)
        out ^= cross & (u < at)
    return out


def _designed(name: str, rng, C):
    """Steps 1 and 3 for a designed valley: the uplift and the rock are laid out by the design
    (designs/<name>.json) - where the valley runs, which peaks stand, which rock is hard, where
    the coast is - and the land is grown from them by the same laws as any other."""
    D = designs().get(name)
    if D is None:
        raise MapError(f"没有这个设计：{name!r}。现有：{', '.join(designs()) or '（designs/ 是空的）'}")
    W, H = D["grid"]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    u, v = (xx + 0.5) / W, (yy + 0.5) / H
    aspect = H / W
    va = v * aspect                                           # distances measured alike across and down
    # the sea: below the coast line - or, for an island, all round the closed coast
    if D.get("island"):
        ragged = D.get("ragged_coast", 0.02) * (_smooth_noise(rng, W, H, H / 9) - 0.5)
        inside = _inside(D["coast"], u, v)
        edge = _to_line([[x, y * aspect] for x, y in D["coast"] + D["coast"][:1]], u, va)
        reach = np.where(inside, edge, -edge) + ragged
        base = reach <= 0
        # the land rises from every shore: low all round the coast, the full uplift inland
        dome = np.clip(reach / D.get("shore", 0.12), 0, 1) ** 0.8
    else:
        base = v > _polyline_v(D["coast"], u)
        dome = 1.0
    # the valley floor runs along the axis; the land rises away from it, and towards the head
    off = _to_line([[x, y * aspect] for x, y in D["axis"]], u, va)
    up = (D.get("floor", 0.15) + D.get("rim", 1.6) * off ** 1.3 + D.get("head", 0.5) * np.clip(1 - v, 0, 1) ** 1.2) * dome
    for b in D.get("rises", []):                              # peaks, shoulders, headlands
        rx, ry = (b["r"], b["r"]) if not isinstance(b["r"], list) else b["r"]
        d2 = ((u - b["at"][0]) / rx) ** 2 + ((v - b["at"][1]) / ry) ** 2
        up = up + b["up"] * np.exp(-d2 * 2.0)
    up *= 1 + D.get("rough", 0.25) * (_smooth_noise(rng, W, H, H / 5) - 0.5)
    hard = np.full((H, W), 0.3, dtype=np.float32)
    for b in D.get("hard", []):
        rx, ry = (b["r"], b["r"]) if not isinstance(b["r"], list) else b["r"]
        d2 = ((u - b["at"][0]) / rx) ** 2 + ((v - b["at"][1]) / ry) ** 2
        hard[d2 + 0.25 * (_smooth_noise(rng, W, H, H / 8) - 0.5) < 1] = 1.0
    h = _grow(up, hard, base, rng, C, D.get("steps", 100), D.get("fill", 2.0))
    # Hand-tuning, after the growing. A bench: the top of an ellipse cut level (at a share of
    # the heights inside it), easing into its surroundings; its edges come out steep.
    for b in D.get("benches", []):
        rx, ry = (b["r"], b["r"]) if not isinstance(b["r"], list) else b["r"]
        d = np.sqrt(((u - b["at"][0]) / rx) ** 2 + ((v - b["at"][1]) / ry) ** 2)
        d = d + b.get("ragged", 0.25) * (_smooth_noise(rng, W, H, H / 10) - 0.5)   # no bench is an ellipse
        inside = d < 1
        if inside.any():
            top = float(np.quantile(h[inside], b.get("level", 0.6)))
            edge = b.get("edge", 0.3)                       # how far (in radii) it takes to ease out: small is a wall
            w = np.clip((1 + edge / 2 - d) / edge, 0, 1)
            h = np.where(inside | (w > 0), h * (1 - w) + top * w, h)
    # A face: along a line, the land on its left (walking from the first point to the next) is
    # held up to the height it has `width` in from the line, so it ends in a wall at the line.
    for f in D.get("faces", []):
        pts = [[x, y * aspect] for x, y in f["line"]]
        dist = _to_line(pts, u, va)
        (ax, ay), (bx, by) = pts[0], pts[-1]
        left = (bx - ax) * (va - ay) - (by - ay) * (u - ax) < 0
        band = left & (dist < f.get("width", 0.05)) & ~base
        if band.any():
            hold = float(np.quantile(h[band], f.get("level", 0.85)))
            h = np.where(band, np.maximum(h, hold * np.clip(1 - dist / f.get("width", 0.05) * 0.15, 0, 1)), h)
    relief_m, cell_m = float(D.get("relief_m", 420)), float(D.get("cell_m", 25))
    land = ~base
    metres = h / max(float(np.quantile(h[land], 0.995)), 1e-9) * relief_m
    relief = float(np.quantile(metres[land], 0.995))
    z = np.where(base, -0.02 - 0.012 * _spread(land, 8), np.maximum(metres, 0.3) / relief)
    return z, hard, z.copy(), relief, cell_m, f"设计的谷地：{D['name']}。{D.get('about', '')}"


def _measured(source: str, seed: int):
    """Steps 1 and 3 as they really happened somewhere: measured heights, brought to the valley
    grid. What the rock is like is read back from the land: where it stands steep, it is hard."""
    S = sources()
    if source not in S:
        raise MapError(f"没有这个真实谷地：{source!r}。现有：{', '.join(S) or '（dem/ 是空的）'}")
    W, H = VALLEY
    raw = np.load(DEM / f"{source}.npy").astype(np.float64)
    raw = np.rot90(raw, S[source].get("turn", 0))        # the sea to the south, as everywhere here
    if seed % 2:
        raw = raw[:, ::-1]                                # a different valley of the same making: its mirror
    img = Image.fromarray(raw.astype(np.float32)).resize((W, H), Image.BILINEAR)
    metres = np.asarray(img, dtype=np.float64)
    cell_m = S[source]["cell_m"] * raw.shape[1] / W

    # Where the sea stands is ours to say (the data is poor there, and this is not that
    # place anyway): high enough to drown the mouth of the valley, a seventh of the map.
    sea, level = _drown(metres, 0.14)
    relief = float(np.quantile(metres[~sea], 0.995) - level)
    z = np.where(sea, -0.02 - 0.012 * _spread(~sea, 8), np.maximum(metres - level, 0.3) / relief)
    metres = np.maximum(metres - level, 0)
    steep = np.degrees(np.arctan(np.hypot(np.gradient(metres, axis=0), np.gradient(metres, axis=1)) / cell_m))
    hard = np.where(_blur(steep, 1.2) > 21, 1.0, 0.3).astype(np.float32)                        # resistant rock stands steep
    return z, hard, z.copy(), relief, cell_m, f"真实地形：{S[source]['name']}。{S[source]['about']}"


@functools.lru_cache(maxsize=8)
def valley(seed: int = 1, climate: str = "oceanic", source: str | None = None, design: str | None = None) -> dict:
    """The whole valley. Its land is grown (`_evolved`), or with `source` a real valley's
    from dem/, to compare; the water on it, what the water leaves, and what grows are worked out here.
    Slow (a few seconds); remembered."""
    if climate not in CLIMATES:
        raise MapError(f"气候只能是 {' / '.join(CLIMATES)}")
    C = CLIMATES[climate]
    rng = np.random.default_rng(seed)
    W, H = VALLEY
    if design:
        z, hard, start, relief_m, cell_m, origin = _designed(design, rng, C)
    else:
        z, hard, start, relief_m, cell_m, origin = _measured(source, seed) if source else _evolved(rng, C)
    H, W = z.shape
    sea, land = z < 0, z >= 0

    # 2 and 4. the water on this land: where it gathers, and what it would cut and drop
    rain = C["rain"] * (0.6 + 1.0 * np.clip(z, 0, None))
    dz, _, _, scree = _work(z, hard, rain)
    laid, cut = np.clip(dz, 0, None), np.clip(-dz, 0, None)
    zf, recv, order, slope = _drainage(z, sea)
    water = rain.ravel().copy()
    rl = recv.ravel()
    for i in order:
        if rl[i] >= 0:
            water[rl[i]] += water[i]
    water = water.reshape(H, W)
    stream_at = float(np.quantile(water[land], 1 - C["streams"]))
    channel = land & (water > stream_at)
    # What a reach can carry goes by its stream power - how steeply it falls times how much water
    # falls (here slope x discharge^0.5) - not by slope alone: a big river on a gentle grade moves
    # stones a steep trickle cannot. Each reach's power, carried out onto its banks.
    power = np.where(channel, slope * np.sqrt(water / stream_at), 0.0)
    reach = power
    for _ in range(3):
        grown = reach.copy()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            grown = np.maximum(grown, np.roll(reach, (dy, dx), axis=(0, 1)))
        reach = np.where(reach > 0, reach, grown)

    # 5. damp, shelter, scour
    wet = np.log1p(_blur(water, 1.3))
    wet = (wet - wet.min()) / max(float(wet.max() - wet.min()), 1e-6)
    shelter = np.zeros((H, W), dtype=np.float32)
    for k in range(1, 9):                                    # something higher upwind (the wind is from the west)
        upwind = np.full((H, W), -9.0)
        upwind[:, k:] = z[:, :-k]
        shelter = np.maximum(shelter, (upwind - z > 0.03 + 0.006 * k).astype(np.float32))
    to_sea = _spread(sea, 14)
    to_water = _spread(channel, 8)
    top = float(np.quantile(z[land], 0.995))
    scour = np.clip(1 - to_sea / 7.0, 0, 1) + 2.4 * np.clip(z / top - 0.6, 0, 1)   # sea wind; and the tops
    damp = (0.6 * wet + 0.3 * np.exp(-to_water / 2.0) + 0.22 * _blur(shelter, 1.5) - 0.42 * scour
            + 0.16 * (_smooth_noise(rng, W, H, H / 6) - 0.5) + 0.08 * (np.gradient(z, axis=0) < 0))
    degrees = np.degrees(np.arctan(np.hypot(np.gradient(z, axis=0), np.gradient(z, axis=1)) * relief_m / cell_m))

    return {"seed": seed, "climate": climate, "source": source, "origin": origin, "start": start, "z": z, "hard": hard,
            "water": water, "recv": recv, "slope": slope, "channel": channel, "stream_at": stream_at, "reach": reach,
            "laid": laid, "grain": reach, "scree": scree, "cut": cut, "damp": damp.astype(np.float32), "wet": wet,
            "top": top, "relief_m": relief_m, "cell_m": cell_m, "degrees": degrees,
            "power": power, "steep": float(np.quantile(power[channel], 0.66)), "gentle": float(np.quantile(power[channel], 0.3)),
            "to_sea": to_sea, "design": design,
            "zoom": designs()[design].get("zoom", ZOOM) if design else ZOOM,
            "sites": designs()[design].get("sites", []) if design else [],
            "cover": designs()[design].get("cover", {}) if design else {},
            "spits": designs()[design].get("spits", []) if design else [],
            "deposits": designs()[design].get("deposits", []) if design else []}


def find(V: dict, what: str, seed: int = 0):
    """A spot in the valley (x, y in valley cells) that shows one thing well. A place made only of
    the one thing - a map that is all cliff - shows nothing, so spots where the window would also
    hold gentle ground score higher. The seed chooses among the best few, far enough apart."""
    if what not in PLACES:
        raise MapError(f"地点只能是 {' / '.join(PLACES)}，或直接给谷地坐标")
    H, W = V["z"].shape
    z, ch, sea = V["z"], V["channel"], V["z"] < 0
    ok = np.zeros((H, W), dtype=bool)
    ok[7:H - 7, 7:W - 7] = True
    grade = np.hypot(np.gradient(z, axis=0), np.gradient(z, axis=1))
    big = np.log1p(V["water"])
    never = -np.inf
    score = {
        "mouth": lambda: np.where(ch & (V["to_sea"] <= 2), big, never),
        "river": lambda: np.where(ch & (V["to_sea"] > 12) & (V["power"] < V["steep"]), big, never),
        # strong for its size (stream power ~ the channel steepness index): where it crosses hard rock
        "gorge": lambda: np.where(ch & (V["hard"] > 0.9) & (V["to_sea"] > 6) & (V["power"] >= V["steep"]), V["power"], never),
        "scarp": lambda: np.where(sea, never, _blur(grade * (V["hard"] > 0.9), 2) + 0.004 * np.exp(-_spread(ch, 8) / 3.0)),
        "bay": lambda: np.where((~sea) & (V["to_sea"] == 2) & (V["hard"] < 0.5), -grade + 0.002 * _spread(ch, 6), never),
        "headland": lambda: np.where((~sea) & (V["to_sea"] <= 2), grade + z + 0.05 * V["hard"], never),   # high, steep ground at the sea
        "upland": lambda: np.where((~sea) & (V["to_sea"] > 13), _blur(z, 2) - 0.3 * np.abs(np.arange(W)[None, :] / W - 0.5), never),
    }[what]()
    deg = V["degrees"]
    gentle = _blur((deg < 15).astype(np.float32), 3.5)     # share of gentle ground around (about a map's width)
    steep = _blur((deg > 32).astype(np.float32), 3.5)
    if what in ("scarp", "gorge", "upland", "headland"):
        score = score / max(float(np.nanmax(np.where(np.isfinite(score), np.abs(score), 0))), 1e-9) \
            + (0.4 if what == "gorge" else 1.2) * np.minimum(gentle, 0.5) - 0.6 * np.clip(steep - 0.6, 0, None)
    score = np.where(ok, score, never)
    picks, flat = [], np.argsort(-score, axis=None)
    for i in flat[:4000].tolist():
        if not np.isfinite(score.flat[i]):
            break
        y, x = divmod(i, score.shape[1])
        if all(abs(x - px) > 10 or abs(y - py) > 7 for px, py in picks):
            picks.append((x, y))
        if len(picks) == 4:
            break
    if not picks:
        raise MapError(f"这个谷地里找不到{PLACES[what]}")
    x, y = picks[(seed // 2) % len(picks)]     # odd seeds mirror the valley; seed // 2 moves along the candidates
    return float(x), float(y)


# ---------------------------------------------------------------- a place in it

def place(width: int, height: int, seed: int = 1, climate: str = "oceanic", at="river",
          source: str | None = None, design: str | None = None) -> dict:
    """A window on the valley at walking scale. `at` is one of PLACES, (x, y) in valley cells,
    or "whole" - the whole valley, which is what a designed valley is for (width and height are
    then its grid times its zoom). Returns {"cells": ground id per cell, "height": level per
    cell, "report": [...], ...}."""
    V = valley(int(seed), climate, source, design)
    C, k = CLIMATES[climate], V["zoom"]
    VH, VW = V["z"].shape
    if at == "whole":
        width, height = VW * k, VH * k
    if not (16 <= width <= mapkit.MAX_SIDE and 12 <= height <= mapkit.MAX_SIDE):
        raise MapError(f"地图宽 16..{mapkit.MAX_SIDE}、高 12..{mapkit.MAX_SIDE}")
    if at == "whole":
        cx, cy = VW / 2 - 0.5, VH / 2 - 0.5
    else:
        cx, cy = find(V, at, int(seed)) if isinstance(at, str) else (float(at[0]), float(at[1]))
    w, h = width / k, height / k
    x0 = min(max(cx + 0.5 - w / 2, 0), VW - w)
    y0 = min(max(cy + 0.5 - h / 2, 0), VH - h)
    rng = np.random.default_rng(int(seed) * 7919 + int(x0 * 13) + int(y0 * 131))

    def field(a):
        """A valley field read `k` times finer over this window."""
        xs = np.clip(x0 + (np.arange(width) + 0.5) / k - 0.5, 0, VW - 1.001)
        ys = np.clip(y0 + (np.arange(height) + 0.5) / k - 0.5, 0, VH - 1.001)
        ix, iy = xs.astype(int), ys.astype(int)
        tx, ty = (xs - ix)[None, :], (ys - iy)[:, None]
        a = a.astype(np.float64)
        top = a[np.ix_(iy, ix)] * (1 - tx) + a[np.ix_(iy, ix + 1)] * tx
        bot = a[np.ix_(iy + 1, ix)] * (1 - tx) + a[np.ix_(iy + 1, ix + 1)] * tx
        return top * (1 - ty) + bot * ty

    z = field(V["z"]) + 0.012 * (_smooth_noise(rng, width, height, 7) - 0.5) + 0.006 * (_smooth_noise(rng, width, height, 3) - 0.5)
    hard = (field(V["hard"]) + 0.25 * (_smooth_noise(rng, width, height, 9) - 0.5)) > 0.65
    damp = field(V["damp"]) + 0.1 * (_smooth_noise(rng, width, height, 11) - 0.5)
    reach, wet, scree = field(V["reach"]), field(V["wet"]), field(V["scree"])
    # how steep, in degrees: the valley's own slope, plus the little ups and downs added at this scale
    grade = np.maximum(field(V["degrees"]), np.degrees(np.arctan(
        np.hypot(np.gradient(z, axis=0), np.gradient(z, axis=1)) * k * V["relief_m"] / V["cell_m"])) * 0.6)
    sea = z < 0

    # the valley's channels, drawn here by the laws a river keeps (see _rivers)
    rv = _rivers(V, x0, y0, k, width, height, rng)
    river, wide = rv["river"], rv["wide"]
    river &= ~sea
    land = ~sea & ~river
    to_river, to_wide, to_sea = _spread(river, 6), _spread(wide & ~sea, 6), _spread(sea, 10)
    mouth = _spread(river & (to_sea <= 1), 8)
    soil = (np.clip(1 - grade / 34.0, 0, 1) * (0.5 + 0.5 * np.where(hard, 0.45, 1.0)) * C["soil"]
            + np.clip(field(V["laid"]) * 6, 0, 0.5))

    names = ["sea", "water", "mud", "sand", "gravel", "grass", "forest", "mountain", "crag", "heath", "scree",
             "platform", "pool", "stack", "cave", "shingle", "saltmarsh", "lagoon", "bog", "bogpool", "flowers", "drygrass", "snow", "copper"]
    (SEA, WATER, MUD, SAND, GRAVEL, GRASS, FOREST, ROCK, CRAG, HEATH, SCREE,
     PLATFORM, POOL, STACK, CAVE, SHINGLE, SALTMARSH, LAGOON, BOG, BOGPOOL) = range(20)
    high = np.clip(z / V["top"], 0, 1)                                        # 0 at the sea .. 1 at the valley's top
    kind = np.full((height, width), GRASS, dtype=np.int8)
    # Open hill: high ground, steep ground, and ground the wind scours carry heath, not meadow.
    cover = V.get("cover", {})
    kind[(high > cover.get("heath_high", 0.32)) | (grade > cover.get("heath_grade", 17)) | (damp < C["wood"] - 0.32)] = HEATH
    # Wood where it is damp, sheltered and not too steep - and only below the tree line.
    kind[(damp > C["wood"]) & (grade < 30) & (high < C["treeline"] + 0.08 * (_smooth_noise(rng, width, height, 9) - 0.5))] = FOREST
    # Bare rock only where nothing can hold: the steepest faces, and hard rock that is nearly so.
    kind[(grade > 44) | (hard & (grade > 37)) | (soil < 0.1)] = ROCK
    # What falls from a face lies below it at its own angle: scree. And on hard steep slopes.
    # Only downhill of it, and not as an even band: it rolls, and some of it rolls further.
    rock = kind == ROCK
    above = np.where(rock, z, -np.inf)                                     # the highest rock within reach
    reach_ = np.rint(1 + 2.5 * k / ZOOM * 3 * _smooth_noise(rng, width, height, 2 * k)).astype(int)
    fell = np.zeros((height, width), dtype=bool)
    for step in range(1, int(reach_.max()) + 1):
        grown = above.copy()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            grown = np.maximum(grown, np.roll(above, (dy, dx), axis=(0, 1)))
        above = grown
        fell |= (above > z + 0.004) & (step <= reach_)
    under = fell & ~rock
    kind[under & ((grade > 20) | (scree > float(np.quantile(V["scree"], 0.75))))] = SCREE
    kind[hard & (grade > 33) & (kind != ROCK)] = SCREE
    # Ground that will not drain: not right by the river - that is the levee, a little higher,
    # coarser, dry - but behind it, in the low hollows of the floodplain (backswamps), in patches.
    hollow = z < _blur(z, 3 * k) - 0.002
    kind[land & (grade < 2.5) & (wet > 0.6) & (to_river >= 2) & (to_river <= 2 + 2 * k) & hollow
         & (_smooth_noise(rng, width, height, 3 * k) > 0.6) & (reach < V["steep"])
         & (damp > C["wood"] - 0.12 * C["marsh"])] = MUD
    # What the river has laid beside itself is not a strip of even width: a point bar on the
    # inside of each bend, nothing on the outside (the cut bank), bars taking turns along the
    # straights (see _rivers). What a bar is made of goes by the stream power of its reach:
    # stones where it is strong, sand, mud where it idles - and finer away from the channel.
    bank = land & rv["bar"]
    bp = rv["power"]
    kind[bank & (bp >= V["steep"])] = GRAVEL
    kind[bank & (bp < V["steep"]) & (bp > V["gentle"])] = SAND
    kind[bank & (bp <= V["gentle"])] = MUD
    back = bank & rv["bar_far"]
    kind[back & (kind == SAND) & (wet > 0.5)] = MUD
    kind[back & (kind == GRAVEL)] = SAND
    flat = np.clip(1 - grade / 14.0, 0, 1)
    kind[land & (to_sea <= np.rint(1 + 6 * np.where(hard, 0.15, 1.0) * flat))] = SAND     # the sea lays sand in the soft bays
    kind[land & (to_sea <= 2) & (grade > 20)] = CRAG                          # and cannot, where the land comes down steep
    kind[land & (mouth <= 6) & (to_sea <= 5) & (grade < 6)] = MUD             # a mouth shelters the water: mud
    kind[river] = WATER
    kind[sea] = SEA
    # Water lies level across and falls smoothly along: a river's surface takes the valley's
    # height without the small ups and downs added to the ground at this scale.
    level = np.where(sea, 0, np.clip(np.ceil(z / V["top"] * LEVELS * FINE), 1, LEVELS * FINE)).astype(int)
    on = river & (rv["level"] >= 0)
    level[on] = np.maximum(rv["level"][on], 1)
    near = np.where(on, level, -1)
    for _ in range(2):                         # the banks at least as high as the water beside them
        grown = near.copy()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            grown = np.maximum(grown, np.roll(near, (dy, dx), axis=(0, 1)))
        near = np.where(on, near, grown)
    bankside = ~sea & ~river & (near > 0)
    level[bankside] = np.maximum(level[bankside], near[bankside])
    coast = _coast_and_bog(kind, level, locals())
    high_ = _high_ground(kind, locals())
    biome, plants, objects, meadow = _biomes(kind, locals())
    meadow = high_ + meadow

    def n(kd):
        return int((kind == kd).sum())

    here = ("整个谷地" if at == "whole" else PLACES[at]) if isinstance(at, str) else f"谷地坐标 ({cx:.0f}, {cy:.0f})"
    lo_l, hi_l = ((int(math.ceil(level[~sea].min() / FINE)), int(math.ceil(level[~sea].max() / FINE)))
                  if (~sea).any() else (0, 0))
    if river.any():
        mean = float(reach[river].mean())
        how = "陡段，河床两侧是卵石" if mean >= V["steep"] else "缓段，两岸是泥" if mean <= V["gentle"] else "中段，两岸是沙"
        river_line = f"河：经过这里的是全谷河段里的{how}（水 {n(WATER)} 格）。"
    else:
        river_line = "河：没有河流经过这里。"
    report = [
        f"位置：{here}。谷地（{VW}×{VH}）里 x {x0:.0f}–{x0 + w:.0f}、y {y0:.0f}–{y0 + h:.0f} 的一块，放大 {k} 倍。"
        f"高度在全谷 {LEVELS} 级里占第 {lo_l}–{hi_l} 级。",
        f"气候：{C['name']}。{C['about']}",
        f"坡度：这里最陡 {float(grade.max()):.0f}°、平均 {float(grade.mean()):.0f}°。高差画成一级级台地，台阶下露出土坎或岩壁。",
        river_line,
        f"堆积：卵石 {n(GRAVEL)} 格、沙 {n(SAND)} 格、泥 {n(MUD)} 格。粗的留在急处，细的留在缓处和河口。",
        f"海岸：软岩处是沙滩，硬岩处是崖岸（{n(CRAG)} 格）。" if sea.any() else "海岸：这里看不到海。",
        f"植被：林地 {n(FOREST)} 格（潮湿、背风、土厚、在林线以下），草甸 {n(GRASS)} 格，"
        f"荒坡 {n(HEATH)} 格（高处、陡处、海风吹刮处）。",
        f"岩石：裸岩 {n(ROCK)} 格（只在最陡处），碎石坡 {n(SCREE)} 格（岩壁下和陡的硬岩坡）。",
    ] + coast + meadow
    return {"cells": [[names[int(v)] for v in row] for row in kind], "height": level.tolist(), "report": report,
            "plants": plants, "objects": objects, "biome": biome,
            "window": (x0, y0, w, h), "valley": V,
            "scale": [V["cell_m"] / k, V["relief_m"] * V["top"] / (LEVELS * FINE)]}   # metres a cell across, a unit high


def _chaikin(pts, times=3):
    """Corner-cutting: the grid's 45 and 90 degree turns become curves; the ends stay put.
    `pts` is an (n, d) array - any columns besides x and y are smoothed along with them."""
    for _ in range(times):
        if len(pts) < 3:
            return pts
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * (len(pts) - 1), pts.shape[1]))
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[:1], mid[1:-1], pts[-1:]])
    return pts


def _rivers(V, x0, y0, k, W, H, rng):
    """The valley's rivers drawn at walking scale, by the laws a river keeps.

    - Course: each reach of the valley's channel network (head or junction to the next junction
      or the sea), its grid corners cut into curves, so tributaries come in at a slant and
      nothing runs along the grid.
    - Width from discharge (hydraulic geometry: width ~ discharge^0.5), never narrowing
      downstream, so a river does not swell into lakes or pinch into threads; it widens where a
      tributary joins.
    - Meanders only where there is a floodplain to meander on and the river is slow: a wave
      along the course about eleven widths long, a little skewed; straight in a steep or
      narrow valley; none at the ends of a reach, so confluences meet.
    - The water surface falls downstream only, and not evenly: it lies level in pools and
      drops at knickpoints - over hard rock, or where the stream is strongest - so the falls
      are where they belong and not at every step of the land.
    - Bars: a point bar on the inside of each bend, sized by how sharp the bend is; the outside
      is the cut bank, bare. Along straighter reaches, bars alternate from side to side.
    Returns cells: river, wide, level (relief units, -1 off the river), power (stream power of
    the reach a cell belongs to, for what its bars are made of), bar (bar cells), bar_far
    (bar cells away from the channel, where it is finer)."""
    VH, VW = V["z"].shape
    ch, recv = V["channel"].ravel(), V["recv"].ravel()
    Q, PW, HD, ZV, DG = (V[n].ravel() for n in ("water", "power", "hard", "z", "degrees"))
    top, steep = V["top"], V["steep"]
    n_don = np.zeros(VH * VW, dtype=int)
    for i in np.flatnonzero(ch):
        r = recv[i]
        if r >= 0 and ch[r]:
            n_don[r] += 1
    reaches = []
    for s0 in np.flatnonzero(ch & (n_don != 1)):          # heads and junctions
        path, i = [int(s0)], int(s0)
        while True:
            r = int(recv[i])
            if r < 0:
                break
            path.append(r)
            if not ch[r] or n_don[r] != 1:
                break
            i = r
        if len(path) >= 2:
            reaches.append(path)
    Qmax = float(Q[ch].max()) if ch.any() else 1.0
    river = np.zeros((H, W), dtype=bool)
    wide = np.zeros((H, W), dtype=bool)
    level = np.full((H, W), np.inf)
    power = np.zeros((H, W))
    bar = np.zeros((H, W), dtype=bool)
    bar_far = np.zeros((H, W), dtype=bool)
    knicks = 0
    for path in reaches:
        vy, vx = np.divmod(np.array(path), VW)
        if vx.max() < x0 - 3 or vx.min() > x0 + W / k + 3 or vy.max() < y0 - 3 or vy.min() > y0 + H / k + 3:
            continue
        cols = np.column_stack([(vx - x0 + 0.5) * k - 0.5, (vy - y0 + 0.5) * k - 0.5,
                                Q[path], PW[path], HD[path], ZV[path], DG[path]])
        pts = _chaikin(cols, 3)
        # resample every half cell
        seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        sl = np.concatenate([[0], np.cumsum(seg)])
        if sl[-1] < 1:
            continue
        ss = np.arange(0, sl[-1], 0.5)
        P = np.column_stack([np.interp(ss, sl, pts[:, j]) for j in range(pts.shape[1])])
        x, y, q, pw, hd, zz, dg = P.T
        w = np.maximum(1.0, 4.5 * (k / 4) * np.sqrt(np.maximum(q, 0) / Qmax))
        w = np.maximum.accumulate(w)                                   # never narrower downstream
        # meanders, on the floodplain only
        free = np.clip((7.0 - dg) / 5.0, 0, 1) * np.clip(1 - 0.7 * pw / max(steep, 1e-9), 0, 1)
        free = _smooth1(free, 9)
        lam = 11.0 * w
        phase = np.cumsum(2 * np.pi * 0.5 / lam) + float(rng.uniform(0, 2 * np.pi))
        S = ss[-1]
        taper = np.clip(ss / (0.6 * lam), 0, 1) * np.clip((S - ss) / (0.6 * lam), 0, 1)
        amp = free * (1.3 * w + 0.8) * taper
        off = amp * (np.sin(phase) + 0.18 * np.sin(3 * phase + 0.6))
        tx, ty = np.gradient(x), np.gradient(y)
        tn = np.maximum(np.hypot(tx, ty), 1e-9)
        nx, ny = -ty / tn, tx / tn
        x, y = x + nx * off, y + ny * off
        # the surface: falls only downstream; level in pools, dropping at knickpoints
        ideal = np.minimum.accumulate(np.clip(zz / top * LEVELS * FINE, 1, LEVELS * FINE))
        lv = ideal.copy()
        cur = np.floor(ideal[0] / 8)
        head = n_don[path[0]] == 0                      # this reach starts at a spring, not a junction
        for i in range(len(ss)):
            t = np.floor(ideal[i] / 8)
            # a fall wants water to go over it: enough coming from upstream, and not at the spring
            fed = q[i] >= 2 * V["stream_at"] and not (head and ss[i] < 4 * k)
            knick = fed and (hd[i] > 0.9 or pw[i] >= steep)
            if t < cur and (knick or cur - t >= 2 or i == len(ss) - 1):
                if knick:
                    knicks += 1
                cur = t
            lv[i] = min(max(ideal[i], cur * 8), cur * 8 + 7)
        # curvature of the course as drawn: which side is the inside of a bend, and how sharp
        tx, ty = np.gradient(x), np.gradient(y)
        ax, ay = np.gradient(tx), np.gradient(ty)
        curv = (tx * ay - ty * ax) / np.maximum(np.hypot(tx, ty), 1e-9) ** 3
        curv = _smooth1(curv, 7)
        tn = np.maximum(np.hypot(tx, ty), 1e-9)
        nx, ny = -ty / tn, tx / tn
        alt = np.sign(np.sin(np.cumsum(2 * np.pi * 0.5 / (6.0 * w)) + float(rng.uniform(0, 6.3))))
        room = np.clip((9.0 - dg) / 6.0, 0, 1)
        for i in range(len(ss) - 1):
            r = max(w[i] / 2, 0.6)
            ax_, ay_, bx_, by_ = x[i], y[i], x[i + 1], y[i + 1]
            lo_x, hi_x = int(np.floor(min(ax_, bx_) - r)), int(np.ceil(max(ax_, bx_) + r))
            lo_y, hi_y = int(np.floor(min(ay_, by_) - r)), int(np.ceil(max(ay_, by_) + r))
            if hi_x < 0 or hi_y < 0 or lo_x >= W or lo_y >= H:
                continue
            lo_x, lo_y, hi_x, hi_y = max(lo_x, 0), max(lo_y, 0), min(hi_x, W - 1), min(hi_y, H - 1)
            gy, gx = np.mgrid[lo_y:hi_y + 1, lo_x:hi_x + 1]
            dx, dy = bx_ - ax_, by_ - ay_
            tt = np.clip(((gx - ax_) * dx + (gy - ay_) * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
            inside = np.hypot(gx - ax_ - tt * dx, gy - ay_ - tt * dy) <= r
            sub_l = level[lo_y:hi_y + 1, lo_x:hi_x + 1]
            sub_l[inside] = np.minimum(sub_l[inside], lv[i])
            river[lo_y:hi_y + 1, lo_x:hi_x + 1] |= inside
            power[lo_y:hi_y + 1, lo_x:hi_x + 1][inside] = pw[i]
            if w[i] >= 3:
                wide[lo_y:hi_y + 1, lo_x:hi_x + 1] |= inside
            # bars: inside of the bend; or, where it runs straight, alternating sides
            bend = float(np.clip(abs(curv[i]) * w[i] * 2.5, 0, 1))
            if bend > 0.15:
                side, size = np.sign(curv[i]), bend * 1.6 * w[i]
            else:
                side, size = alt[i], 0.6 * w[i] * float(_hash01(i, len(path)) > 0.45)
            size *= room[i]
            o = r + 0.4
            while o <= r + size:
                cx, cy = int(round(x[i] + nx[i] * side * o)), int(round(y[i] + ny[i] * side * o))
                if 0 <= cx < W and 0 <= cy < H:
                    bar[cy, cx] = True
                    power[cy, cx] = pw[i]
                    if o > r + 0.55 * size:
                        bar_far[cy, cx] = True
                o += 0.5
    level = np.where(np.isfinite(level), np.rint(level), -1).astype(int)
    return {"river": river, "wide": wide, "level": level, "power": power, "bar": bar & ~river, "bar_far": bar_far & ~river,
            "knicks": knicks}


def _smooth1(a, n):
    """A running mean along a line, as long as the line is long enough for it."""
    return np.convolve(a, np.ones(n) / n, mode="same") if len(a) >= n else a


def _hash01(a, b):
    v = math.sin(a * 12.9898 + b * 78.233) * 43758.5453
    return v - math.floor(v)


def _coast_and_bog(kind, level, at):
    """The shore and the wet tops, after the rest of the ground is laid; changes `kind` and
    `level` in place, returns lines for the report. `at` is place()'s locals.

    Rock coast: the sea planes a shelf at the foot of a hard coast (a wave-cut platform), leaves
    pools in it, leaves stacks standing off it where the cliff fell back, and cuts caves into the
    foot of the tallest faces. Soft coast: where the shore lies open to the waves, storms throw up
    ridges of shingle; sheltered bays keep their sand. A spit (from the design) shuts a lagoon in
    behind it. By a river mouth or a lagoon, low flat sheltered ground is salt marsh, drained by
    branching creeks. On the flat wet tops, peat: blanket bog, with pools that lie along the
    contours."""
    sea, land, river, hard, grade, z = at["sea"], at["land"], at["river"], at["hard"], at["grade"], at["z"]
    rng, V, C, k, H, W = at["rng"], at["V"], at["C"], at["k"], at["height"], at["width"]
    x0, y0, VW, VH = at["x0"], at["y0"], at["VW"], at["VH"]
    (SEA, WATER, MUD, SAND, GRAVEL, GRASS, FOREST, ROCK, CRAG, HEATH, SCREE,
     PLATFORM, POOL, STACK, CAVE, SHINGLE, SALTMARSH, LAGOON, BOG, BOGPOOL) = range(20)
    lines = []
    salt = float(rng.uniform(0, 100))
    if sea.any():
        open_ = _blur(sea.astype(np.float32), 3.0 * k)                    # how much sea a spot faces: bays are low
        to_sea = _spread(sea, 12)
        # -- rock coast
        cliffy = land & hard & (to_sea <= 1 + k // 2)                    # hard rock meeting the sea
        d_rock = _spread(cliffy, 10)
        shelf = sea & (d_rock <= np.rint(1.5 + 0.9 * k * _smooth_noise(rng, W, H, 2 * k)))
        kind[shelf] = PLATFORM
        level[shelf] = 1
        inner = shelf & (_spread(sea & ~shelf, 3) >= 2) & (_spread(land, 2) >= 2)
        pools = inner & (_smooth_noise(rng, W, H, max(2, k * 0.8)) > 0.6)
        kind[pools] = POOL
        # stacks: a few, just off the shelf, where the cliff once was
        ring = sea & ~shelf & (d_rock >= 2) & (d_rock <= 3 + k)
        cand = np.argwhere(ring)
        stacks = 0
        if len(cand):
            rng.shuffle(cand)
            top = np.where(cliffy, level, 0).max() if cliffy.any() else LEVELS * FINE // 2
            placed = []
            for y, x in cand:
                if stacks >= 2 + int(cliffy.sum() // (60 * k * k)) or stacks >= 7:
                    break
                if any(abs(x - px) + abs(y - py) < 6 * k for px, py in placed):
                    continue
                r = 1.2 + 0.4 * k * float(rng.random())
                yy, xx = np.ogrid[:H, :W]
                body = ((xx - x) ** 2 + (yy - y) ** 2 <= r * r) & sea & ~shelf
                if not body.any():
                    continue
                kind[body] = STACK
                level[body] = int(top * (0.45 + 0.3 * rng.random()))
                foot = (_spread(body, 1) == 1) & sea & ~shelf          # a skirt of shelf round its foot
                kind[foot] = PLATFORM
                level[foot] = 1
                placed.append((x, y))
                stacks += 1
        # caves: at the foot of the tallest faces, opening onto the shelf or the sea
        foot = cliffy & ((_spread(shelf, 1) == 1) | (_spread(sea, 1) == 1))
        caves = 0
        if foot.any():
            tall = np.where(foot, _blur(level.astype(np.float32) * cliffy, 2.0), -1)
            for _ in range(2 if cliffy.sum() > 40 * k * k else 1 if cliffy.sum() > 8 * k else 0):
                y, x = np.unravel_index(int(np.argmax(tall)), tall.shape)
                if tall[y, x] <= 0:
                    break
                size = max(1, k // 2)
                kind[max(0, y - size + 1):y + 1, max(0, x - size + 1):x + 1] = CAVE
                tall[max(0, y - 8 * k):y + 8 * k, max(0, x - 8 * k):x + 8 * k] = -1
                caves += 1
        # -- soft coast: shingle where it is open, in ridges along the shore
        shore = land & (to_sea <= 2)
        exposed = land & ~hard & ~river & (to_sea <= 2 + k) & (open_ > float(np.quantile(open_[shore], 0.8))
                                                                if shore.any() else False)
        ridge = exposed & ((to_sea <= 1 + k // 2) | ((to_sea >= 2 + k) & (to_sea <= 2 + k) & (grade < 8)))
        ridge &= np.isin(kind, [SAND, GRASS, HEATH, MUD])
        kind[ridge] = SHINGLE
        # -- spits from the design, and the lagoons behind them
        lagoons = 0
        for sp in V.get("spits", []):
            pts = [((u * VW - x0) * k - 0.5, (v * VH - y0) * k - 0.5) for u, v in sp["line"]]
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
            dist = np.full((H, W), np.inf)
            for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                dx, dy = bx - ax, by - ay
                t = np.clip(((xx - ax) * dx + (yy - ay) * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
                dist = np.minimum(dist, np.hypot(xx - ax - t * dx, yy - ay - t * dy))
            width = sp.get("width", 0.6) * k * (0.8 + 0.4 * _smooth_noise(rng, W, H, 3 * k))
            bar = (dist <= width) & (kind == SEA)
            if not bar.any():
                continue
            # what of the sea is cut off from the open water by the bar is the lagoon
            water = (kind == SEA) & ~bar
            outside = np.zeros_like(water)
            outside[0, :], outside[-1, :], outside[:, 0], outside[:, -1] = water[0, :], water[-1, :], water[:, 0], water[:, -1]
            while True:
                grown = outside.copy()
                grown[1:, :] |= outside[:-1, :]
                grown[:-1, :] |= outside[1:, :]
                grown[:, 1:] |= outside[:, :-1]
                grown[:, :-1] |= outside[:, 1:]
                grown &= water
                if (grown == outside).all():
                    break
                outside = grown
            shut = water & ~outside
            kind[bar] = SAND
            level[bar] = 1
            kind[shut] = LAGOON
            # an inlet at the free end keeps the lagoon open to the sea
            ex, ey = pts[-1]
            inlet = (np.hypot(xx - ex, yy - ey) <= max(1.2, 0.5 * k)) & bar
            kind[inlet] = SEA
            level[inlet] = 0
            lagoons += int(shut.any())
        # -- salt marsh by the mouths and lagoons, with creeks
        still = (kind == LAGOON) | (river & (at["mouth"] <= 2 * k))
        near_still = _spread(still, 3 * k)
        marsh = (land & ~river & (level <= max(2, FINE // 3)) & (grade < 4) & (near_still <= 2 * k)
                 & ((at["mouth"] <= 4 * k) | (_spread(kind == LAGOON, 2 * k) <= 2 * k))
                 & np.isin(kind, [MUD, SAND, GRASS, HEATH, FOREST]) & (open_ < 0.62))
        kind[marsh] = SALTMARSH
        creeks = 0
        if marsh.any():
            wet = np.isin(kind, [SEA, LAGOON, WATER])
            dw = _spread(wet, 30).astype(np.float32)
            starts = np.argwhere(marsh & (_spread(wet, 1) == 1))
            rng.shuffle(starts)
            for sy, sx in starts[: max(1, int(marsh.sum() // (150 * k)))]:
                todo = [(int(sy), int(sx), 3 * k + int(rng.integers(0, 4 * k)))]
                while todo:
                    y, x, left = todo.pop()
                    while left > 0:
                        kind[y, x] = WATER
                        best = None
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W and kind[ny, nx] == SALTMARSH:
                                score = dw[ny, nx] + 1.4 * float(rng.random())     # inland, with a wander
                                if best is None or score > best[0]:
                                    best = (score, ny, nx)
                        if best is None:
                            break
                        _, y, x = best
                        left -= 1
                        if rng.random() < 0.05 and left > 2 * k:
                            todo.append((y, x, left // 2))
                    creeks += 1
        lines.append(f"海岸：硬岩岸前是海蚀平台（{int((kind == PLATFORM).sum())} 格，潮池 {int((kind == POOL).sum())} 格），"
                     f"海蚀柱 {stacks} 根，海蚀洞 {caves} 处；迎浪的软岸是卵石脊（{int((kind == SHINGLE).sum())} 格），"
                     f"避风的湾里留着沙。" + (f"沙嘴围出潟湖 {lagoons} 处。" if lagoons else "")
                     + (f"河口与潟湖边的低平地是盐沼（{int((kind == SALTMARSH).sum())} 格），有 {creeks} 条潮沟。" if marsh.any() else ""))
    # -- blanket bog on the flat wet tops, its pools lying along the contours
    high = np.clip(z / V["top"], 0, 1)
    damp = at["damp"]
    # the tops are where the rain falls most; wind dries the sod but not the peat under it
    bog = (land & ~river & (high > 0.28) & (grade < 4 + 2.5 * C["marsh"]) & (damp > C["wood"] - 0.75 * C["marsh"])
           & np.isin(kind, [GRASS, HEATH, FOREST]) & (C["marsh"] >= 0.8))
    bog = (_blur(bog.astype(np.float32), 1.2 * k) > 0.5) & land & ~river & np.isin(kind, [GRASS, HEATH, FOREST])
    if bog.any():
        kind[bog] = BOG
        zs = _blur(z, 2.0 * k)
        gy, gx = np.gradient(zs)
        ang = np.arctan2(gy, gx)                      # the way down; pools lie across it
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        along = (-xx * np.sin(ang) + yy * np.cos(ang)) / (2.6 * k)      # along the contour: long
        down = (xx * np.cos(ang) + yy * np.sin(ang)) / (0.7 * k)        # down the slope: narrow
        pools = bog & (_vnoise(along, down, salt) > 0.74) & (grade < 4) & (_spread(~bog, 2) >= 2)
        kind[pools] = BOGPOOL
        lines.append(f"高处：平缓潮湿的山顶长成泥炭沼（{int(bog.sum())} 格），沼池顺着等高线排开（{int(pools.sum())} 格）。")
    return lines


def _meadows(kind, at):
    """Which meadow each piece of open grass is, and how long its grass grows. Changes `kind`
    in place; returns (report lines, plant id per cell or None).

    - Dry grassland (yellow) where the ground dries out: it drains fast (steeper, thin soil),
      faces the sun (falls away to the south), or the sea wind dries it. More of it in a dry
      climate, little in a cold wet one.
    - Flower meadow where it is rich, sunny and moist but not wet, in patches.
    - Green meadow everywhere else.
    Grass grows long where the soil is deep and damp and nothing scours it; it stays short on
    thin, exposed, windswept ground. Woods, heath, bog and bare ground carry no such grass."""
    (SEA, WATER, MUD, SAND, GRAVEL, GRASS, FOREST, ROCK, CRAG, HEATH, SCREE,
     PLATFORM, POOL, STACK, CAVE, SHINGLE, SALTMARSH, LAGOON, BOG, BOGPOOL, FLOWERS, DRYGRASS, SNOW, COPPER) = range(24)
    z, grade, damp, soil, rng, k = at["z"], at["grade"], at["damp"], at["soil"], at["rng"], at["k"]
    H, W, C = at["height"], at["width"], at["C"]
    grassy = kind == GRASS
    plants = [[None] * W for _ in range(H)]
    if not grassy.any():
        return [], plants
    gy = np.gradient(_blur(z, 1.5 * k), axis=0)
    sun = np.tanh(-gy / (float(np.std(gy)) + 1e-9))                 # +1 falling away south, -1 north
    to_sea = _spread(at["sea"], 4 * k)
    wind = np.clip(1 - to_sea / (3.0 * k), 0, 1)
    dryness = (-damp + 0.25 * sun + 0.012 * grade + 0.3 * (1 - np.clip(soil, 0, 1)) + 0.25 * wind
               + 0.15 * (_smooth_noise(rng, W, H, 4 * k) - 0.5))
    share = {"oceanic": 0.25, "dry": 0.6, "cold": 0.12}.get(at["climate"], 0.25)
    dry = grassy & (dryness > float(np.quantile(dryness[grassy], 1 - share)))
    kind[dry] = DRYGRASS
    rich = (grassy & ~dry & (sun > -0.1) & (damp > C["wood"] - 0.3) & (damp < C["wood"] + 0.05) & (grade < 12)
            & (_smooth_noise(rng, W, H, 3 * k) > 0.58))
    # a flowering patch has no edge on a line: it frays out into the plain meadow around it
    rich = (_blur(rich.astype(np.float32), 1.5 * k) + 0.45 * (_smooth_noise(rng, W, H, max(2, k)) - 0.5) > 0.3) & grassy & ~dry
    kind[rich] = FLOWERS
    meadow = np.isin(kind, [GRASS, FLOWERS, DRYGRASS])
    lush = np.clip(soil, 0, 1.2) + 0.6 * damp - 0.5 * wind - 0.015 * grade + 0.25 * (_smooth_noise(rng, W, H, 2 * k) - 0.5)
    long_ = meadow & (lush > float(np.quantile(lush[meadow], 0.5))) & (grade < 18)
    for y, x in zip(*np.nonzero(meadow)):
        if kind[y, x] == FLOWERS:      # the flowers grow in the grass: green meadow, flowering grass on it
            plants[y][x] = "flowergrass_long" if long_[y, x] else "flowergrass"
            continue
        d = kind[y, x] == DRYGRASS
        plants[y][x] = ("longgrass" if long_[y, x] else "shortgrass") + ("_dry" if d else "")
    flowering = kind == FLOWERS
    kind[flowering] = GRASS
    n = lambda v: int((kind == v).sum())  # noqa: E731
    lines = [f"草地：绿草地 {n(GRASS)} 格，其中开花的 {int(flowering.sum())} 格（肥沃、向阳、润而不湿），荒草地 {n(DRYGRASS)} 格"
             f"（排水快、向阳、海风吹干处）；长草 {int(long_.sum())} 格（土厚、潮湿、不受风），其余是短草。"]
    return lines, plants


def _cover(kind, at, plants) -> list:
    """What grows on the grounds other than meadow, each where it would: fills `plants` where
    nothing grows yet, returns report lines.

    - Wood floor: ferns where it is damp and shaded, moss where it is wetter; bare litter between.
    - Heath: heather; bracken on the deeper soil of the slopes; gorse where it is dry and sunny.
    - Sand: marram on the dunes behind the beach, never on the beach the waves reach.
    - Mud: reeds right by still or slow water, rushes a little further off.
    - Salt marsh: samphire by the water and the creeks, sea lavender on the higher marsh.
    - Bog: cotton grass and sphagnum.
    - Rock (scree, ledges, the sea cliff): lichen; moss on damp scree.
    - Shingle: thrift, out of the waves' reach."""
    (SEA, WATER, MUD, SAND, GRAVEL, GRASS, FOREST, ROCK, CRAG, HEATH, SCREE,
     PLATFORM, POOL, STACK, CAVE, SHINGLE, SALTMARSH, LAGOON, BOG, BOGPOOL, FLOWERS, DRYGRASS, SNOW, COPPER) = range(24)
    damp, soil, grade, z, rng, k = at["damp"], at["soil"], at["grade"], at["z"], at["rng"], at["k"]
    H, W, C = at["height"], at["width"], at["C"]
    to_sea = _spread(at["sea"], 4 * k)
    water = np.isin(kind, [WATER, LAGOON, BOGPOOL, POOL])
    to_water = _spread(water, 3 * k)
    gy = np.gradient(_blur(z, 1.5 * k), axis=0)
    sun = np.tanh(-gy / (float(np.std(gy)) + 1e-9))
    free = np.array([[p is None for p in row] for row in plants])
    counts = {}

    def patch(scale):
        return _smooth_noise(rng, W, H, max(2, scale * k))

    def put(mask, name):
        mask = mask & free
        for y, x in zip(*np.nonzero(mask)):
            plants[y][x] = name
        free[mask] = False
        counts[name] = counts.get(name, 0) + int(mask.sum())

    wood = C["wood"]
    high = np.clip(z / at["V"]["top"], 0, 1)
    alpine_at = SNOW_LINE.get(at["climate"], 0.86) - 0.18
    put(np.isin(kind, [SCREE, ROCK]) & (high > alpine_at) & (patch(1) > 0.4), "alpine")
    put((kind == SAND) & (to_sea >= 2 + k // 2) & (patch(2) > 0.35), "marram")
    put((kind == MUD) & (to_water <= 1 + k // 2) & (patch(2) > 0.3), "reeds")
    put((kind == MUD) & (to_water <= 3 * k) & (patch(2) > 0.4), "rushes")
    put((kind == SALTMARSH) & (to_water <= k), "samphire")
    put((kind == SALTMARSH) & (patch(2) > 0.45), "sealavender")
    put((kind == BOG) & (patch(2) > 0.5), "cottongrass")
    put((kind == BOG) & (patch(1) > 0.35), "sphagnum")
    put((kind == SCREE) & (damp > wood) & (patch(2) > 0.55), "moss")
    put(np.isin(kind, [SCREE, ROCK, CRAG]) & (patch(2) > 0.55), "lichen")
    put((kind == SHINGLE) & (to_sea >= 2) & (patch(2) > 0.6), "thrift")
    names = {"moss": "苔藓", "fernbed": "蕨丛", "gorse": "荆豆", "bracken": "欧洲蕨", "heather": "石南", "marram": "滨草",
             "reeds": "芦苇", "rushes": "灯心草", "samphire": "盐角草", "sealavender": "海薰衣草", "cottongrass": "羊胡子草",
             "sphagnum": "泥炭藓", "lichen": "地衣", "thrift": "海石竹", "alpine": "高山垫状植物"}
    grown = "，".join(f"{names[n]} {c} 格" for n, c in counts.items() if c)
    return [f"植被：{grown}。"] if grown else []


# ---------------------------------------------------------------- biomes
#
# After Minecraft: a few smooth fields over the land - how warm (cooler with height, warmer on
# the sunny side, a set warmth for the climate), how wet, how far from the sea, how steep - and
# a table that turns where a place sits in them into a biome. A biome is a recipe: which ground
# it lays on the open land, what grows on it (in patches), which trees in what mix and how close,
# and the odd bush and stone. The grounds that geology and water make - rock, scree, sand, mud,
# the shore, bog, snow, ore - keep their own laws, and their biome follows from them.
#
# plants: [(plant id or None, share)] - laid in patches, so a moor is heather here, bracken
# there, not a speckle of both. trees: {id: share}, density trees per cell. ground: the open
# land's ground in this biome.
BIOMES = {
    "plains":        {"name": "草原", "ground": "grass", "plants": [("shortgrass", 0.5), ("longgrass", 0.4), ("flowergrass", 0.1)],
                      "trees": {"oak": 0.7, "birch": 0.3}, "density": 0.006, "rocks": 0.003},
    "flower_meadow": {"name": "花海", "ground": "grass", "plants": [("flowergrass", 0.6), ("flowergrass_long", 0.3), ("shortgrass", 0.1)],
                      "trees": {"birch": 1.0}, "density": 0.004, "bushes": 0.01},
    "dry_plains":    {"name": "干草原", "ground": "drygrass", "plants": [("shortgrass_dry", 0.6), ("longgrass_dry", 0.35), (None, 0.05)],
                      "trees": {"windbent": 1.0}, "density": 0.004, "rocks": 0.005},
    "forest":        {"name": "阔叶林", "ground": "forest", "plants": [("fernbed", 0.35), (None, 0.5), ("moss", 0.15)],
                      "trees": {"oak": 0.8, "birch": 0.2}, "density": 0.11, "bushes": 0.01},
    "birch_forest":  {"name": "白桦林", "ground": "grass", "plants": [("longgrass", 0.5), ("flowergrass_long", 0.2), ("fernbed", 0.15), (None, 0.15)],
                      "trees": {"birch": 0.9, "oak": 0.1}, "density": 0.1},
    "dark_forest":   {"name": "幽暗林", "ground": "forest", "plants": [("moss", 0.45), ("fernbed", 0.35), (None, 0.2)],
                      "trees": {"oak": 0.55, "alder": 0.45}, "density": 0.2},
    "taiga":         {"name": "针叶林", "ground": "forest", "plants": [("moss", 0.35), (None, 0.45), ("fernbed", 0.2)],
                      "trees": {"pine": 0.85, "birch": 0.15}, "density": 0.12, "rocks": 0.005},
    "moor":          {"name": "荒原", "ground": "heath", "plants": [("heather", 0.55), ("bracken", 0.25), ("gorse", 0.15), (None, 0.05)],
                      "trees": {"windbent": 1.0}, "density": 0.004, "rocks": 0.008},
    "alpine_meadow": {"name": "高山草甸", "ground": "grass", "plants": [("shortgrass", 0.45), ("alpine", 0.4), ("flowergrass", 0.15)],
                      "trees": {}, "density": 0.0, "rocks": 0.012},
    "riverside":     {"name": "河岸", "ground": "grass", "plants": [("longgrass", 0.6), ("flowergrass_long", 0.2), ("rushes", 0.2)],
                      "trees": {"alder": 1.0}, "density": 0.07},
    "swamp":         {"name": "沼泽", "ground": "grass", "plants": [("rushes", 0.5), ("longgrass", 0.3), ("reeds", 0.2)],
                      "trees": {"alder": 0.7, "birch": 0.3}, "density": 0.05},
    # those that follow from the ground
    "beach":         {"name": "海滩"}, "stony_shore": {"name": "石岸"}, "saltmarsh": {"name": "盐沼"},
    "peat_bog":      {"name": "泥炭沼"}, "stony_peaks": {"name": "裸岩"}, "snowy_slopes": {"name": "雪坡"},
    "river":         {"name": "河"}, "sea": {"name": "海"}, "lagoon": {"name": "潟湖"},
}
WARMTH = {"oceanic": 0.85, "dry": 0.95, "cold": 0.55}   # how warm at the sea, before height takes it away
WETTER = {"oceanic": 0.0, "dry": -0.22, "cold": 0.05}


def _biomes(kind, at):
    """The biome of every cell, and what its recipe grows: returns (biome id per cell, plant id
    per cell or None, object id per cell or None, report lines). Changes `kind` on the open land
    (meadow, wood, heath) to the ground of its biome."""
    (SEA, WATER, MUD, SAND, GRAVEL, GRASS, FOREST, ROCK, CRAG, HEATH, SCREE,
     PLATFORM, POOL, STACK, CAVE, SHINGLE, SALTMARSH, LAGOON, BOG, BOGPOOL, FLOWERS, DRYGRASS, SNOW, COPPER) = range(24)
    code = {"grass": GRASS, "forest": FOREST, "heath": HEATH, "drygrass": DRYGRASS}
    z, damp, soil, grade, rng, k, V, C = at["z"], at["damp"], at["soil"], at["grade"], at["rng"], at["k"], at["V"], at["C"]
    H, W = at["height"], at["width"]
    climate = at["climate"]
    high = np.clip(z / V["top"], 0, 1)
    gy = np.gradient(_blur(z, 1.5 * k), axis=0)
    sun = np.tanh(-gy / (float(np.std(gy)) + 1e-9))
    to_sea = _spread(at["sea"], 4 * k)
    wind = np.clip(1 - to_sea / (3.0 * k), 0, 1)
    to_river = _spread(at["river"], 3 * k)
    open_land = np.isin(kind, [GRASS, FOREST, HEATH])
    # the fields, smooth over a few dozen steps so that biomes come in whole regions
    warm = _blur(WARMTH.get(climate, 0.85) - 0.9 * high + 0.08 * sun, 2.5 * k) + 0.06 * (_smooth_noise(rng, W, H, 8 * k) - 0.5)
    land = ~at["sea"] & ~at["river"]
    rank = np.zeros((H, W))
    if land.any():
        order = np.argsort(np.argsort(damp[land]))
        rank[land] = order / max(len(order) - 1, 1)
    wet = _blur(rank, 2.5 * k) + WETTER.get(climate, 0.0) + 0.08 * (_smooth_noise(rng, W, H, 8 * k) - 0.5)
    shelter = 1 - wind
    choice = np.full((H, W), "", dtype=object)
    cold, cool = warm < 0.25, (warm >= 0.25) & (warm < 0.45)
    choice[cold & (wet < 0.55)] = "alpine_meadow"
    choice[cold & (wet >= 0.55)] = "taiga"
    choice[cool & (wet < 0.45)] = "moor"
    choice[cool & (wet >= 0.45)] = "taiga"
    mild = warm >= 0.45
    choice[mild & (wet < 0.22)] = "dry_plains"
    choice[mild & (wet >= 0.22) & (wet < 0.45)] = "plains"
    choice[mild & (wet >= 0.45) & (wet < 0.72)] = "forest"
    choice[mild & (wet >= 0.72)] = "dark_forest"
    # turns within the table: sunny rich grass flowers; the windward edge of a wood is birch, and
    # the sea wind keeps the coast open; steep slopes in the cool are moor
    flowers = (choice == "plains") & (sun > 0) & (_smooth_noise(rng, W, H, 5 * k) > 0.55)
    choice[flowers] = "flower_meadow"
    choice[(choice == "forest") & ((wind > 0.3) | (_smooth_noise(rng, W, H, 6 * k) > 0.68))] = "birch_forest"
    choice[(choice == "dark_forest") & (shelter < 0.6)] = "forest"
    choice[np.isin(choice, ["forest", "birch_forest", "dark_forest", "plains"]) & (wind > 0.5) & (sun < 0.3)] = "moor"
    choice[(grade > 24) & np.isin(choice, ["plains", "flower_meadow", "dry_plains"])] = "moor"
    choice[(to_river <= 1 + k) & mild & (grade < 10)] = "riverside"
    choice[(kind == MUD) & (to_river > 1 + k)] = "swamp"
    # whole regions: each cell takes the biome most of its neighbourhood has
    names_ = [b for b in BIOMES if BIOMES[b].get("ground")]
    best, out = np.full((H, W), -1.0), np.full((H, W), "", dtype=object)
    for b in names_:
        m = choice == b
        if not m.any():
            continue
        f = _blur(m.astype(np.float32), 1.2 * k) + 0.15 * m
        win = f > best
        best[win], out[win] = f[win], b
    biome = np.where(open_land | (kind == MUD), out, "")
    # the ground decides it elsewhere
    biome[np.isin(kind, [SAND])] = "beach"
    biome[np.isin(kind, [SHINGLE, PLATFORM, POOL, STACK, CAVE, CRAG])] = "stony_shore"
    biome[kind == SALTMARSH] = "saltmarsh"
    biome[np.isin(kind, [BOG, BOGPOOL])] = "peat_bog"
    biome[np.isin(kind, [ROCK, SCREE, COPPER, GRAVEL])] = "stony_peaks"
    biome[kind == SNOW] = "snowy_slopes"
    biome[kind == MUD] = np.where(biome[kind == MUD] == "", "swamp", biome[kind == MUD])
    biome[kind == WATER] = "river"
    biome[kind == LAGOON] = "lagoon"
    biome[kind == SEA] = "sea"
    biome[biome == ""] = "plains"
    # the open land's ground is its biome's
    for b in names_:
        g = BIOMES[b]["ground"]
        kind[open_land & (biome == b)] = code[g]
    # what grows: each biome's plants in patches (the same field everywhere, so patches run
    # on across a border)
    plants = [[None] * W for _ in range(H)]
    patch = _smooth_noise(rng, W, H, max(2, 2 * k))
    patch = (np.argsort(np.argsort(patch.ravel())) / patch.size).reshape(H, W)   # even 0..1
    long_ok = (grade < 18)
    for b in names_:
        here = (biome == b) & np.isin(kind, [GRASS, FOREST, HEATH, DRYGRASS, MUD])
        if not here.any():
            continue
        acc = 0.0
        for pid, share in BIOMES[b]["plants"]:
            sel = here & (patch >= acc) & (patch < acc + share)
            acc += share
            if pid is None:
                continue
            if pid in ("longgrass", "longgrass_dry", "flowergrass_long"):
                short = {"longgrass": "shortgrass", "longgrass_dry": "shortgrass_dry", "flowergrass_long": "flowergrass"}[pid]
                for y, x in zip(*np.nonzero(sel)):
                    plants[y][x] = pid if long_ok[y, x] else short
            else:
                for y, x in zip(*np.nonzero(sel)):
                    plants[y][x] = pid
        # by the water within a biome: reeds at the edge of the mud
    lines = []
    lines += _cover(kind, at, plants)
    objects, wood = _trees(kind, at, plants, biome)
    count = {}
    for b in biome.ravel().tolist():
        count[b] = count.get(b, 0) + 1
    lines.insert(0, "生物群系：" + "，".join(f"{BIOMES[b]['name']} {n}" for b, n in sorted(count.items(), key=lambda kv: -kv[1])
                                         if b not in ("sea",)) + " 格。")
    return [list(row) for row in biome], plants, objects, lines + wood


SNOW_LINE = {"oceanic": 0.86, "dry": 0.93, "cold": 0.62}   # share of the valley's height above which snow lies all year


def _high_ground(kind, at) -> list:
    """Snow and ore: changes `kind`, returns report lines.

    Snow lies all year above the snow line - lower in a cold climate - and longest where the
    slope faces away from the sun; it does not hold on faces steeper than about 40 degrees, and
    it is not on water. Ore shows where the design puts a deposit and the rock is hard and bare
    enough to show it (an outcrop, not under soil)."""
    (SEA, WATER, MUD, SAND, GRAVEL, GRASS, FOREST, ROCK, CRAG, HEATH, SCREE,
     PLATFORM, POOL, STACK, CAVE, SHINGLE, SALTMARSH, LAGOON, BOG, BOGPOOL, FLOWERS, DRYGRASS, SNOW, COPPER) = range(24)
    z, grade, rng, k, V = at["z"], at["grade"], at["rng"], at["k"], at["V"]
    H, W = at["height"], at["width"]
    high = np.clip(z / V["top"], 0, 1)
    gy = np.gradient(_blur(z, 1.5 * k), axis=0)
    sun = np.tanh(-gy / (float(np.std(gy)) + 1e-9))
    line = SNOW_LINE.get(at["climate"], 0.86) + 0.05 * sun + 0.04 * (_smooth_noise(rng, W, H, 3 * k) - 0.5)
    snow = (high > line) & (grade < 40) & np.isin(kind, [GRASS, HEATH, FOREST, SCREE, ROCK, BOG, DRYGRASS, FLOWERS, CRAG])
    kind[snow] = SNOW
    lines = [f"高处：雪线在全谷高度的 {SNOW_LINE.get(at['climate'], 0.86) * 100:.0f}% 左右（背阴坡更低），"
             f"终年积雪 {int(snow.sum())} 格；太陡处存不住雪。"] if snow.any() else []
    ore = 0
    for dep in V.get("deposits", []):
        VH, VW = V["z"].shape
        cx, cy = (dep["at"][0] * VW - at["x0"]) * k - 0.5, (dep["at"][1] * VH - at["y0"]) * k - 0.5
        rx, ry = dep["r"] if isinstance(dep["r"], list) else (dep["r"], dep["r"])
        yy, xx = np.mgrid[0:H, 0:W]
        inside = (((xx - cx) / (rx * VW * k)) ** 2 + ((yy - cy) / (ry * VH * k)) ** 2
                  + 0.5 * (_smooth_noise(rng, W, H, max(2, k)) - 0.5) < 1)
        show = inside & at["hard"] & np.isin(kind, [ROCK, SCREE, HEATH, GRASS, DRYGRASS, CRAG])
        kind[show] = COPPER
        ore += int(show.sum())
    if ore:
        lines.append(f"矿：东山脚的硬岩露头带着铜绿和锈色（{ore} 格）。")
    return lines


def _trees(kind, at, plants, biome=None):
    """Trees, shrubs and boulders, as objects: returns (object id per cell or None, report lines).

    Trees stand in the woods - closer where the soil is deep and damp - and one here and there on
    the meadow. Which tree is a matter of where: alder by the water; pine high up, on hard rock and
    on thin soil; birch at the wood's edge, where wood is taking new ground; trees bent low by the
    wind near the tree line and where the sea wind blows; oak in the sheltered lowland. Berry
    bushes line the wood's edge. Boulders lie on the heath and the scree, fallen or left there.
    No tree is put where it would shut a cell in: one can always walk between them."""
    (SEA, WATER, MUD, SAND, GRAVEL, GRASS, FOREST, ROCK, CRAG, HEATH, SCREE,
     PLATFORM, POOL, STACK, CAVE, SHINGLE, SALTMARSH, LAGOON, BOG, BOGPOOL, FLOWERS, DRYGRASS, SNOW, COPPER) = range(24)
    z, damp, soil, grade, rng, k, V, C = at["z"], at["damp"], at["soil"], at["grade"], at["rng"], at["k"], at["V"], at["C"]
    H, W, hard = at["height"], at["width"], at["hard"]
    high = np.clip(z / V["top"], 0, 1)
    tl = C["treeline"]
    to_sea = _spread(at["sea"], 4 * k)
    wind = np.clip(1 - to_sea / (3.0 * k), 0, 1)
    to_river = _spread(at["river"] | (kind == LAGOON), 3 * k)
    forest = kind == FOREST
    edge = forest & (_spread(~forest & ~at["sea"] & ~at["river"], 2) <= 1 + k // 4)
    noise = _smooth_noise(rng, W, H, 3 * k)
    kind_of = np.full((H, W), "oak", dtype=object)
    per_cell = np.zeros((H, W))
    rocks = np.zeros((H, W))
    bushes = np.zeros((H, W))
    if biome is not None:
        # each biome's own mix and closeness; which of its trees stands where by a smooth field
        bio = np.array(biome, dtype=object)
        pick = _smooth_noise(rng, W, H, 2 * k)
        for b, rec in BIOMES.items():
            here = bio == b
            if not here.any() or not rec.get("trees"):
                rocks[here] = rec.get("rocks", 0)
                continue
            acc = 0.0
            for tid, share in rec["trees"].items():
                kind_of[here & (pick >= acc) & (pick < acc + share + (1 if acc + share >= 0.999 else 0))] = tid
                acc += share
            per_cell[here] = rec.get("density", 0)
            rocks[here] = rec.get("rocks", 0)
            bushes[here] = rec.get("bushes", 0)
    else:
        kind_of[edge & (noise > 0.45)] = "birch"
        kind_of[(high > tl - 0.12) | hard | (soil < 0.35)] = "pine"
    kind_of[(high > tl - 0.04) | (wind > 0.55)] = np.where(kind_of[(high > tl - 0.04) | (wind > 0.55)] == "pine", "pine", "windbent")
    kind_of[to_river <= 1 + k // 2] = "alder"
    walk = ~np.isin(kind, [SEA, WATER, LAGOON, POOL, BOGPOOL, STACK]) & ~at["river"]
    blocked = ~walk
    objects = [[None] * W for _ in range(H)]
    chance = rng.random((H, W))
    # a tree's crown spans two or three cells, so a tree to every five or six cells already closes
    # the canopy; less where the soil is thin, with clearings where the noise says
    closeness = np.clip(0.07 + 0.15 * (damp - C["wood"]) + 0.08 * np.clip(soil, 0, 1), 0.04, 0.2) * (
        0.5 + _smooth_noise(rng, W, H, 4 * k))

    def free_ways(y, x):
        return sum(1 for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                   if 0 <= y + dy < H and 0 <= x + dx < W and not blocked[y + dy, x + dx])

    def place(y, x, tid):
        # never shut a neighbour in: each walkable neighbour keeps two ways out
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not blocked[ny, nx] and free_ways(ny, nx) - 1 < 2:
                return False
        objects[y][x] = tid
        blocked[y, x] = True
        return True

    counts = {}
    outside_wood = _spread(forest, 1) == 1          # the first cell out from the wood
    order = np.argsort(chance, axis=None)
    for i in order.tolist():
        y, x = divmod(i, W)
        if blocked[y, x]:
            continue
        tid = None
        if biome is not None:
            soft = kind[y, x] in (GRASS, FOREST, HEATH, DRYGRASS, MUD)
            if soft and chance[y, x] < per_cell[y, x] * (0.5 + noise[y, x]) and grade[y, x] < 38 and high[y, x] < tl + 0.05:
                tid = kind_of[y, x]
            elif soft and kind[y, x] != FOREST and outside_wood[y, x] and chance[y, x] < 0.04 + bushes[y, x]:
                tid = "berry_bush"                                                 # the wood's edge
            elif walk[y, x] and kind[y, x] not in (SAND, MUD, SALTMARSH, SHINGLE) and chance[y, x] < rocks[y, x] + (
                    0.02 if kind[y, x] == SCREE else 0):
                tid = "rock"
        elif forest[y, x] and chance[y, x] < closeness[y, x] and grade[y, x] < 38:
            tid = kind_of[y, x]
        elif kind[y, x] in (GRASS, DRYGRASS) and chance[y, x] < 0.006 and high[y, x] < tl:
            tid = "oak" if damp[y, x] > C["wood"] - 0.2 else "birch"          # a lone tree in the meadow
        elif kind[y, x] == HEATH and chance[y, x] < 0.008 and high[y, x] < tl + 0.05:
            tid = "windbent"
        elif kind[y, x] in (GRASS, HEATH) and outside_wood[y, x] and chance[y, x] < 0.08:
            tid = "berry_bush"                                                 # the wood's edge
        elif kind[y, x] in (HEATH, SCREE, GRASS, DRYGRASS) and chance[y, x] < (0.03 if kind[y, x] == SCREE else 0.004):
            tid = "rock"
        if tid and place(y, x, tid):
            counts[tid] = counts.get(tid, 0) + 1
    names = {"oak": "橡树", "pine": "松", "birch": "白桦", "alder": "桤木", "windbent": "风压树", "berry_bush": "浆果丛", "rock": "大石"}
    line = "树与石：" + "，".join(f"{names[t]} {n}" for t, n in sorted(counts.items(), key=lambda kv: -kv[1])) + \
           "。河边桤木，高处和硬岩上是松，林缘白桦，林线和迎海风处是风压树，避风低地是橡树；林缘有浆果丛。"
    return objects, [line] if counts else []


def connect(m, tiles: dict, cells_min: int = 4) -> dict:
    """See that one can walk everywhere. Cliffs and rivers cut land into pieces; join every piece
    worth the name to the largest, cheapest way first: a ramp up a cliff where it is lowest, or a
    ford where a river is narrowest (and it is narrow where it runs fast over stones). Changes
    the map in place; returns how many of each it made."""
    W, H = m.width, m.height
    ground = m.layers["ground"]
    made = {"ramps": 0, "fords": 0, "cleared": 0}
    objects = m.layers["object"]

    def walk(x, y):
        g = tiles.get(ground[y][x])
        return bool(g) and g.get("pass") == "walk"

    def wet(x, y):
        g = tiles.get(ground[y][x])
        return bool(g) and g.get("pass") == "swim" and g["look"].get("flows")   # rivers, not the sea

    for _ in range(60):
        region, sizes, main = mapkit.land_regions(m, tiles)
        cut_off = {i for i, (n, _) in enumerate(sizes) if i != main and n >= cells_min}
        if main < 0 or not cut_off:
            break
        walls = mapkit.cliffs(m, tiles)
        best = None   # (cost, kind, data)
        for x, y, dx, dy in walls:
            a, b = region[y][x], region[y + dy][x + dx]
            if {a, b} & {main} and ({a, b} - {main}) & cut_off:
                cost = abs(m.relief[y][x] - m.relief[y + dy][x + dx])
                if best is None or cost < best[0]:
                    best = (cost, "ramp", (x, y, dx, dy))
        for y in range(H):
            for x in range(W):
                if region[y][x] != main:
                    continue
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    run, nx, ny = [], x + dx, y + dy
                    while 0 <= nx < W and 0 <= ny < H and wet(nx, ny) and len(run) < 6:
                        run.append((nx, ny))
                        nx, ny = nx + dx, ny + dy
                    if run and 0 <= nx < W and 0 <= ny < H and region[ny][nx] in cut_off:
                        cost = 3 + 4 * len(run)
                        if best is None or cost < best[0]:
                            best = (cost, "ford", (run, dx, dy))
        # or a tree, a bush or a stone in the way: the cheapest of all, it just is not put there
        for y in range(H):
            for x in range(W):
                t = tiles.get(objects[y][x]) if objects[y][x] else None
                if not t or t.get("pass") != "block" or not walk(x, y):
                    continue
                near = {region[ny][nx] for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                        if 0 <= nx < W and 0 <= ny < H}
                if main in near and near & cut_off and (best is None or best[0] > 1):
                    best = (1, "clear", (x, y))
        if best is None:
            break
        if best[1] == "clear":
            x, y = best[2]
            objects[y][x] = None
            made["cleared"] += 1
        elif best[1] == "ramp":
            x, y, dx, dy = best[2]
            m.ramps.append([x, y, dx, dy])
            side = (dy, dx)                                    # widen it by one cell along the cliff, if the cliff goes on
            for s in (1, -1):
                e = (x + side[0] * s, y + side[1] * s, dx, dy)
                if e in walls:
                    m.ramps.append(list(e))
                    break
            made["ramps"] += 1
        else:
            run, dx, dy = best[2]
            across = (dy, dx)
            for cx, cy in run:                                 # two cells wide, so it reads as a crossing
                ground[cy][cx] = "ford"
                nx, ny = cx + across[0], cy + across[1]
                if 0 <= nx < W and 0 <= ny < H and wet(nx, ny):
                    ground[ny][nx] = "ford"
            made["fords"] += 1
    return made


def make_map(project: mapkit.Project, name: str, width: int, height: int, seed: int = 1, climate: str = "oceanic",
             at="river", source: str | None = None, design: str | None = None) -> dict:
    """Work out a place and save it as a map. Fails if the tileset lacks a ground it calls for."""
    out = place(width, height, seed, climate, at, source, design)
    width, height = len(out["cells"][0]), len(out["cells"])
    missing = sorted({t for row in out["cells"] for t in row} - set(project.tiles()))
    if missing:
        raise MapError(f"图块集里缺少这些地形：{', '.join(missing)}")
    project.create_map(name, width, height)

    def fill(m, tiles):
        m.layers["ground"] = [list(row) for row in out["cells"]]
        m.relief = [list(row) for row in out["height"]]
        m.scale = list(out["scale"])
        if "plant" in m.layers and all(t in tiles for t in {p for row in out["plants"] for p in row if p}):
            m.layers["plant"] = [list(row) for row in out["plants"]]
        if all(t in tiles for t in {o for row in out["objects"] for o in row if o}):
            m.layers["object"] = [list(row) for row in out["objects"]]
        m.biome = [list(row) for row in out["biome"]]
        return connect(m, tiles)

    made = project.edit(name, fill)["result"]
    if made["ramps"] or made["fords"] or made.get("cleared"):
        out["report"].append(f"通行：只有坡度超过 {mapkit.CLIFF_DEG:.0f}° 的地方才是崖；为了让整张地图走得通，在崖上开了 {made['ramps']} 处坡道，"
                             f"在河上设了 {made['fords']} 处浅滩（河最窄、水最急最浅的地方）"
                             + (f"，挪掉了 {made['cleared']} 棵把路完全堵死的树或石头。" if made.get("cleared") else "。"))
    else:
        out["report"].append(f"通行：只有坡度超过 {mapkit.CLIFF_DEG:.0f}° 的地方才是崖，整张地图本来就走得通。")
    out["connect"] = made
    return out


def account(V: dict) -> list:
    """What the valley is and what happened to it, in a few lines."""
    C, land = CLIMATES[V["climate"]], V["z"] >= 0
    deg = V["degrees"][land]
    return [
        f"地形：{V['origin']}",
        f"尺度：谷地格网 {V['z'].shape[1]}×{V['z'].shape[0]}，每格约 {V['cell_m']:.0f} 米，高差约 {V['relief_m']:.0f} 米。"
        f"平地（<5°）占 {float((deg < 5).mean()) * 100:.0f}%，陡坡（>30°）占 {float((deg > 30).mean()) * 100:.0f}%。",
        f"气候：{C['name']}。{C['about']}",
        f"水文：{int(V['channel'].sum())} 格是溪与河，支流汇入干流后入海。",
        "堆积：河段越陡留下的越粗——陡段卵石、中段沙、缓段与河口是泥；平缓的海岸积沙，陡岸是崖。",
    ]


def pictures(V: dict, scale: int = 5, window=None) -> dict:
    """The valley's fields as pictures, for seeing what each step did. Not maps to play on."""
    def ramp(a, lo, hi, stops):
        t = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
        rgb = np.zeros(a.shape + (3,), dtype=np.float32)
        pos = np.linspace(0, 1, len(stops))
        for c in range(3):
            rgb[..., c] = np.interp(t, pos, [s[c] for s in stops])
        return rgb

    def show(rgb):
        rgb = np.clip(rgb, 0, 255).copy()
        if window:
            x0, y0, w, h = window
            a, b, c, d = int(x0), int(y0), int(x0 + w), int(y0 + h)
            for yy in (b, d):
                rgb[min(max(yy, 0), rgb.shape[0] - 1), max(a, 0):c + 1] = (255, 240, 120)
            for xx in (a, c):
                rgb[max(b, 0):d + 1, min(max(xx, 0), rgb.shape[1] - 1)] = (255, 240, 120)
        img = Image.fromarray(rgb.astype(np.uint8))
        return img.resize((img.width * scale, img.height * scale), Image.NEAREST)

    land_ramp = [(60, 84, 66), (120, 136, 92), (176, 170, 124), (150, 140, 130), (232, 230, 224)]
    sea_col = np.array([37, 63, 72], dtype=np.float32)

    def relief(z):
        rgb = ramp(z, 0, float(z.max()), land_ramp)
        light = 1 + np.clip((np.gradient(z, axis=1) + np.gradient(z, axis=0)) * -7, -0.4, 0.4)   # lit from the top-left
        rgb = rgb * light[..., None]
        rgb[z < 0] = sea_col
        return rgb

    z = V["z"]
    hard = np.where(V["hard"][..., None] > 0.9, np.array([98, 101, 110]), np.array([176, 160, 132])).astype(np.float32)
    hard[V["start"] < 0] = sea_col
    water = relief(z) * 0.55
    flow = np.log1p(V["water"])
    base = math.log1p(V["stream_at"]) * 0.82
    t = np.clip((flow - base) / max(float(flow.max()) - base, 1e-9), 0, 1)[..., None] ** 0.7
    water = water * (1 - t) + np.array([150, 205, 235]) * t
    water[z < 0] = sea_col
    damp = ramp(V["damp"], float(np.quantile(V["damp"], 0.03)), float(np.quantile(V["damp"], 0.97)),
                [(196, 180, 130), (150, 160, 100), (70, 110, 80), (30, 62, 58)])
    damp[z < 0] = sea_col
    laid = relief(z) * 0.5
    d = np.clip(V["laid"] / max(float(np.quantile(V["laid"], 0.99)), 1e-9), 0, 1)[..., None]
    grain = np.clip(V["grain"] / max(V["steep"] * 1.6, 1e-9), 0, 1)[..., None]   # fast water left it: stones; slow: mud
    laid = laid * (1 - d) + (np.array([122, 92, 70]) * (1 - grain) + np.array([226, 222, 210]) * grain) * d
    laid[z < 0] = sea_col
    return {"start": show(relief(V["start"])), "hard": show(hard), "height": show(relief(z)), "water": show(water),
            "laid": show(laid), "damp": show(damp)}
