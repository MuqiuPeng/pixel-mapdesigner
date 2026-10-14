"""The trees, drawn the way pixel artists draw them, not as blobs.

Broadleaf crowns (oak, alder, birch, the wind-bent tree) follow the leaf-bundle method
(SLYNYRD, Pixelblog 44 "Top Down Trees"): one small bundle of leaves - 2x2 leaves round a
centre, shaded like a ball lit from the top left - is made in a dark, a middle and a light
version; the crown is built from many bundles laid back to front, the upper-left ones light,
the lower-right and lowest ones dark; each bundle drops a shadow on the ones below it; stray
pixels are cleaned away; the crown's edge gets the darkest green, not ink (a selective outline,
so a soft shape is not cut out with scissors). The trunk is a cylinder lit from the left that
flares into roots, with bark streaks down its middle and a knot; a branch or two shows through
the gaps low in the crown.

Conifers (pine) are built branch tier by branch tier, top to bottom: each tier a drooping skirt,
lit on the left and dark on the right, with the shadow of the tier above lying on its top.

Everything is seeded: the same call draws the same tree. Colours are palette names.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image

T = 0  # transparent in the index grids below

LEAF = {
    "oak": ["g0", "g1", "g2", "g3", "g4"],
    "alder": ["g0", "g0", "g1", "g2", "g3"],
    "birch": ["g1", "g2", "g3", "g4", "g4"],
}
BARK = {
    "oak": ["n0", "w0", "w1", "w2"],
    "alder": ["n0", "w0", "w0", "w1"],
    "birch": ["s1", "s3", "s4", "f0"],
}


def _bundle(rng, r=4.2):
    """One bundle of leaves: offsets and a shade (0 dark .. 2 light) for each pixel. Leaves are
    2x2 squares on a staggered grid inside a ball; the gaps between them are a step darker, so
    the bundle reads as leaves and not as a disc; the ball is lit from the top left."""
    px = {}
    for gy in range(-5, 6, 2):
        for gx in range(-5, 6, 2):
            ox = gx + (1 if (gy // 2) % 2 else 0)
            for dy in (0, 1):
                for dx in (0, 1):
                    x, y = ox + dx, gy + dy
                    d = math.hypot(x + 0.5, (y + 0.5) * 1.15)
                    if d > r + rng.uniform(-0.4, 0.3):
                        continue
                    light = -(x * 0.65 + y * 0.85) / r          # toward the top left
                    shade = 2 if light > 0.38 else 0 if light < -0.3 else 1
                    if dx == 1 and dy == 1:                      # the lower-right pixel of a leaf: its own shadow
                        shade = max(shade - 1, 0)
                    px[(x, y)] = shade
    # the gaps inside the ball: a step darker than what is around them
    for y in range(-6, 7):
        for x in range(-6, 7):
            if (x, y) not in px and math.hypot(x + 0.5, (y + 0.5) * 1.15) < r - 1.2:
                px[(x, y)] = 0
    return px


def _clean(idx):
    """Single pixels that match none of their four neighbours take the commonest of them."""
    h, w = idx.shape
    out = idx.copy()
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            v = idx[y, x]
            if v == T:
                continue
            n = [idx[y - 1, x], idx[y + 1, x], idx[y, x - 1], idx[y, x + 1]]
            if v not in n and all(c != T for c in n):
                out[y, x] = max(set(n), key=n.count)
    return out


def _outline(idx, ink):
    """A one-pixel edge round the shape, in `ink`."""
    filled = idx != T
    edge = np.zeros_like(filled)
    edge[1:, :] |= filled[:-1, :]
    edge[:-1, :] |= filled[1:, :]
    edge[:, 1:] |= filled[:, :-1]
    edge[:, :-1] |= filled[:, 1:]
    out = idx.copy()
    out[edge & ~filled] = ink
    return out


def _image(idx, names, P):
    """Index grid (0 transparent, 1.. into `names`) to an image."""
    h, w = idx.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for i, n in enumerate(names, start=1):
        rgba[idx == i] = P[n]
    return Image.fromarray(rgba)


def _trunk(idx, cx, top, bottom, width, bark, rng, lean=0.0):
    """A trunk from `top` to `bottom` (rows), `width` at the foot, narrowing a little upward,
    flaring into roots at the foot, lit from the left; bark streaks and a knot. `bark` are
    indices dark .. light (4)."""
    dark, mid_d, mid, light = bark
    for y in range(top, bottom + 1):
        t = (y - top) / max(bottom - top, 1)
        half = width / 2 * (0.75 + 0.25 * t)
        if y >= bottom - 1:
            half += 1.5 if y == bottom else 0.8                 # the roots spread
        x0 = cx + lean * (bottom - y) - half
        xs = range(int(round(x0)), int(round(x0 + 2 * half)))
        for i, x in enumerate(xs):
            if not 0 <= x < idx.shape[1]:
                continue
            u = i / max(len(xs) - 1, 1)
            idx[y, x] = light if u < 0.25 else mid if u < 0.6 else mid_d if u < 0.85 else dark
    # bark: broken streaks down the middle, a knot
    for _ in range(max(2, (bottom - top) // 4)):
        y = rng.randint(top, bottom - 2)
        x = int(round(cx + lean * (bottom - y))) + rng.choice([-1, 0, 0, 1])
        for k in range(rng.randint(2, 3)):
            if 0 <= y + k < idx.shape[0] and idx[y + k, x] != T:
                idx[y + k, x] = mid_d
    ky = rng.randint(top + 2, max(top + 3, bottom - 4))
    kx = int(round(cx + lean * (bottom - ky)))
    if idx[ky, kx] != T:
        idx[ky, kx] = dark


def broadleaf(P, kind="oak", w=40, h=54, crown=(0.5, 0.36, 0.46, 0.34), trunk_w=5, seed=0, bundles=26, lean=0.0,
              squash=1.0):
    """A broadleaf tree `w` x `h` px. crown: centre x, centre y, half-width, half-height as
    shares of w and h. The trunk stands at the bottom centre."""
    rng = random.Random(f"{kind}:{seed}")
    leaf, bark = LEAF[kind], BARK[kind]
    names = leaf + bark                        # indices 1..5 leaves dark..light, 6..9 bark dark..light
    L = [1, 2, 3, 4, 5]
    B = [6, 7, 8, 9]
    idx = np.zeros((h, w), dtype=np.int16)
    ccx, ccy, rx, ry = crown[0] * w, crown[1] * h, crown[2] * w, crown[3] * h * squash
    cx = w / 2 - 0.5
    _trunk(idx, cx, int(ccy), h - 2, trunk_w, B, rng, lean)
    if kind == "birch":   # a birch's white bark is marked across with short black dashes
        for y in range(int(ccy) + 2, h - 3, 3):
            x = int(round(cx + lean * (h - 2 - y))) + rng.choice([-1, 0])
            for k in range(rng.randint(1, 2)):
                if 0 <= x + k < w and idx[y, x + k] != T:
                    idx[y, x + k] = B[0]
    # a couple of branches reaching into the crown, seen through its gaps low down
    for side in (-1, 1):
        bx, by = cx, ccy + ry * 0.35
        for k in range(int(rx * 0.55)):
            x, y = int(round(bx + side * k)), int(round(by - k * 0.6))
            if 0 <= x < w and 0 <= y < h:
                idx[y, x] = B[1]
    # the bundles: placed in the crown, spread by a little rejection, sorted back to front
    pts = []
    tries = 0
    while len(pts) < bundles and tries < 4000:
        tries += 1
        a, d = rng.uniform(0, 2 * math.pi), math.sqrt(rng.random())
        x, y = ccx + math.cos(a) * d * rx * 0.86, ccy + math.sin(a) * d * ry * 0.86
        if all(math.hypot(x - px, y - py) > 4.2 for px, py in pts):
            pts.append((x, y))
    pts.sort(key=lambda p: p[1])
    variant = []
    for x, y in pts:                          # where it sits against the light decides its value
        side = ((x - ccx) / rx) * 0.6 + ((y - ccy) / ry) * 0.8
        variant.append(0 if side > 0.35 else 2 if side < -0.25 else 1)
    shape = _bundle(rng)
    for (x, y), v in zip(pts, variant):
        # its shadow on what is already there below and to the right
        for (ox, oy), s in shape.items():
            sx, sy = int(round(x + ox + 1)), int(round(y + oy + 2))
            if 0 <= sx < w and 0 <= sy < h and idx[sy, sx] in L and idx[sy, sx] > L[0]:
                idx[sy, sx] -= 1
        for (ox, oy), s in shape.items():
            px_, py_ = int(round(x + ox)), int(round(y + oy))
            if 0 <= px_ < w and 0 <= py_ < h:
                lvl = min(max(s + v - 1, 0), 3)     # 0 deepest .. 3 lightest of the four leaf values
                idx[py_, px_] = L[lvl]
    # a few highlights on the most lit bundles
    for (x, y), v in zip(pts, variant):
        if v == 2 and rng.random() < 0.7:
            for ox, oy in ((-2, -2), (-1, -2), (-2, -1)):
                px_, py_ = int(round(x + ox)), int(round(y + oy))
                if 0 <= px_ < w and 0 <= py_ < h and idx[py_, px_] in L:
                    idx[py_, px_] = L[4]
    idx = _clean(idx)
    # selective outline: the crown's edge in its darkest green, the trunk's in ink
    crown_mask = np.isin(idx, L)
    out = _outline(np.where(crown_mask, idx, T), L[0])
    out_trunk = _outline(np.where(np.isin(idx, B), idx, T), B[0])
    idx = np.where((out == L[0]) & (idx == T), L[0], idx)
    idx = np.where((out_trunk == B[0]) & (idx == T), B[0], idx)
    return _image(idx, names, P)


def conifer(P, w=30, h=56, tiers=7, seed=0):
    """A pine: tiers of drooping branches from the top down, each lit on the left, the shadow of
    the one above lying across its top; a short trunk under the lowest."""
    rng = random.Random(f"pine:{seed}")
    names = ["g0", "g1", "g2", "g3", "n0", "w0", "w1", "w2"]
    D, M, Lt, H = 1, 2, 3, 4                  # leaf: deepest, dark, mid, light (g0 g1 g2 g3)
    B = [5, 6, 7, 8]
    idx = np.zeros((h, w), dtype=np.int16)
    cx = w / 2 - 0.5
    _trunk(idx, cx, h - 12, h - 2, 4, B, rng)
    top, bottom = 1, h - 9
    step = (bottom - top) / tiers
    for i in range(tiers):
        y0 = top + i * step * 0.92
        th = step * 1.45
        half_w = 2 + (w / 2 - 2) * ((i + 1) / tiers) ** 0.85
        for yy in range(int(y0), int(y0 + th) + 1):
            t = (yy - y0) / th
            hw = half_w * min(1.0, max(t, 0.0) * 1.6) ** 0.9
            droop = 0
            for xx in range(int(cx - hw), int(cx + hw) + 2):
                if not 0 <= xx < w or not 0 <= yy < h:
                    continue
                u = (xx - cx) / max(hw, 1)
                # the skirt droops: its outer ends hang lower
                if t > 0.62 + 0.38 * (1 - abs(u) ** 2) or (t > 0.85 and abs(u) < 0.5):
                    continue
                lit = -u * 0.8 - (t - 0.4) * 0.6
                v = H if lit > 0.55 else Lt if lit > 0.05 else M if lit > -0.45 else D
                idx[yy, xx] = v
        # the shadow of this tier on the top of the next
        ys = int(y0 + th * 0.75)
        for yy in range(ys, min(ys + 2, h)):
            for xx in range(w):
                if idx[yy, xx] in (Lt, H) and (xx - cx) > -half_w * 0.3:
                    idx[yy, xx] = M
    # needle texture: a few staggered dark marks and light tips
    for _ in range(int(w * h * 0.03)):
        x, y = rng.randrange(w), rng.randrange(h)
        if idx[y, x] in (Lt, M) and idx[y, max(x - 1, 0)] != T:
            idx[y, x] = idx[y, x] - 1
    idx = _clean(idx)
    out = _outline(np.where(np.isin(idx, [D, M, Lt, H]), idx, T), D)
    idx = np.where((out == D) & (idx == T), D, idx)
    out_t = _outline(np.where(np.isin(idx, B), idx, T), B[0])
    idx = np.where((out_t == B[0]) & (idx == T), B[0], idx)
    return _image(idx, names, P)


def oak(P, seed=0):
    return broadleaf(P, "oak", 42, 54, (0.5, 0.36, 0.47, 0.34), trunk_w=6, seed=seed, bundles=30)


def alder(P, seed=0):
    return broadleaf(P, "alder", 32, 50, (0.5, 0.38, 0.42, 0.38), trunk_w=4, seed=seed, bundles=22, squash=1.05)


def birch(P, seed=0):
    return broadleaf(P, "birch", 26, 50, (0.5, 0.3, 0.42, 0.28), trunk_w=3, seed=seed, bundles=13)


def windbent(P, seed=0):
    """A tree the west wind shapes: low, the trunk leaning east, the crown pushed east and flat."""
    return broadleaf(P, "oak", 38, 32, (0.62, 0.36, 0.38, 0.3), trunk_w=4, seed=seed, bundles=16, lean=-0.35, squash=0.85)


def pine(P, seed=0):
    return conifer(P, 30, 56, 7, seed)
