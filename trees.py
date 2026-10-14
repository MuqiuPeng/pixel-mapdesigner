"""The trees, drawn big shapes first, the way pixel artists draw them.

Broadleaf crowns (oak, alder, birch, the wind-bent tree):

1. The crown is a few lobes - not one ball, not many little bundles - each with a scalloped,
   leafy rim. Lower lobes stand in front of higher ones.
2. Light comes from the top left. Each point is lit as part of its own lobe (a ball) and as part
   of the whole crown (a bigger ball); a lobe in front throws a shadow on the lobe behind it, and
   the crown is darkest underneath, where the leaves keep the sun off the trunk.
3. That light is cut into five tones. Leaf texture is added only near the cuts - where one tone
   turns into the next - so the large areas stay clean and the crown reads as a volume, not as
   noise. (Pixel Joint critiques: "the canopy is too detailed"; shade the sphere first, detail in
   some areas only.)
4. Shadows are cooler, lights warmer (the palette's ramps shift hue); the rim of the crown is
   its own darkest green, softened on the lit side - not ink.
5. The trunk tapers up from flaring roots, is lit from the left, lies in the crown's shadow at
   the top, and forks into branches that show in the notch between the lowest lobes.

Conifers (pine) are built tier by tier from the top: each tier a skirt of drooping branches
with ragged tips, lit on the left, with a band of shadow on the tier below it.

Everything is seeded: the same call draws the same tree. Colours are palette names.
References: SLYNYRD, Pixelblog 15 "Plant Life" and 44 "Top Down Trees"; Pixel Joint forum
critique threads on tree shading.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image

T = 0  # transparent in the index grids below

# five tones of leaf, deepest .. lightest; four of bark, darkest .. lit
LEAF = {
    "oak": ["g0", "g1", "g2", "g3", "g4"],
    "alder": ["b0", "g0", "g1", "g2", "g3"],
    "birch": ["g1", "g2", "g3", "g4", "g4"],
    "pine": ["b0", "g0", "g1", "g2", "g3"],
}
BARK = {
    "oak": ["n0", "w0", "w1", "w2"],
    "alder": ["n0", "w0", "w1", "w1"],
    "birch": ["n0", "s3", "s4", "f0"],
    "pine": ["n0", "w0", "w1", "w2"],
}


def _hash(x, y, salt):
    v = np.sin(x * 127.1 + y * 311.7 + salt * 74.7) * 43758.5453
    return v - np.floor(v)


def _shift(a, oy, ox, fill=False):
    """`a` moved by (oy, ox) without wrapping round the edges."""
    out = np.full_like(a, fill)
    h, w = a.shape
    out[max(oy, 0):h + min(oy, 0), max(ox, 0):w + min(ox, 0)] = a[max(-oy, 0):h + min(-oy, 0), max(-ox, 0):w + min(-ox, 0)]
    return out


def _image(idx, names, P):
    """Index grid (0 transparent, 1.. into `names`) to an image."""
    h, w = idx.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for i, n in enumerate(names, start=1):
        rgba[idx == i] = P[n]
    return Image.fromarray(rgba)


def _clean(idx, keep=()):
    """Single pixels that match none of their four neighbours take the commonest of them."""
    h, w = idx.shape
    out = idx.copy()
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            v = idx[y, x]
            if v == T or v in keep:
                continue
            n = [idx[y - 1, x], idx[y + 1, x], idx[y, x - 1], idx[y, x + 1]]
            if v not in n and all(c != T for c in n):
                out[y, x] = max(set(n), key=n.count)
    return out


def _trunk(idx, cx, top, bottom, width, bark, rng, lean=0.0, birch=False):
    """A trunk from row `top` down to `bottom`: tapering up from flaring roots, lit from the
    left; bark streaks (or, on a birch, short dark dashes across) and a knot."""
    dark, mid_d, mid, light = bark
    for y in range(top, bottom + 1):
        t = (y - top) / max(bottom - top, 1)
        half = width / 2 * (0.7 + 0.3 * t)
        if y >= bottom - 2:
            half += (y - (bottom - 3)) * 0.9          # the roots spread
        x0 = cx + lean * (bottom - y) - half
        xs = list(range(int(round(x0)), int(round(x0 + 2 * half))))
        for i, x in enumerate(xs):
            if 0 <= x < idx.shape[1]:
                u = i / max(len(xs) - 1, 1)
                idx[y, x] = light if u < 0.28 else mid if u < 0.6 else mid_d if u < 0.88 else dark
    if birch:
        for y in range(top + 2, bottom - 2, 3):
            x = int(round(cx + lean * (bottom - y))) + rng.choice([-1, 0])
            for k in range(rng.randint(1, 2)):
                if idx[y, x + k] != T:
                    idx[y, x + k] = dark
    else:
        for _ in range(max(2, (bottom - top) // 4)):
            y = rng.randint(top, bottom - 3)
            x = int(round(cx + lean * (bottom - y))) + rng.choice([-1, 0, 1])
            for k in range(rng.randint(2, 3)):
                if idx[y + k, x] not in (T, light):
                    idx[y + k, x] = mid_d
        ky = rng.randint(top + 2, max(top + 3, bottom - 5))
        kx = int(round(cx + lean * (bottom - ky))) + 1
        if idx[ky, kx] != T:
            idx[ky, kx] = dark


def broadleaf(P, kind="oak", w=42, h=56, lobes=6, crown=(0.5, 0.36, 0.44, 0.33), trunk_w=6, seed=0, lean=0.0,
              stretch=1.0, gaps=0):
    """A broadleaf tree `w` x `h` px. crown: centre x, centre y, half-width, half-height as
    shares of w and h. `stretch` > 1 draws the crown out to the east (the wind-bent tree)."""
    rng = random.Random(f"{kind}:{seed}")
    leaf, bark = LEAF[kind], BARK[kind]
    names = leaf + bark
    L = [1, 2, 3, 4, 5]                    # leaf tones, deepest .. lightest
    B = [6, 7, 8, 9]                        # bark, darkest .. lit
    idx = np.zeros((h, w), dtype=np.int16)
    ccx, ccy, rx, ry = crown[0] * w, crown[1] * h, crown[2] * w, crown[3] * h
    cx = w / 2 - 0.5
    # the trunk and its branches first: the crown covers most of them
    tb = int(ccy + ry * 0.55)
    _trunk(idx, cx, int(ccy + ry * 0.1), h - 2, trunk_w, B, rng, lean, birch=(kind == "birch"))
    trunk_m = idx != T
    branch = np.zeros((h, w), dtype=bool)
    for side in (-1, 1):
        for k in range(int(rx * 0.6)):
            x, y = int(round(cx + side * k * 0.9 + lean * (h - 2 - tb))), int(round(tb - k * 0.75))
            for dx in (0, side):
                if 0 <= x + dx < w and 0 <= y < h and not trunk_m[y, x + dx]:
                    idx[y, x + dx] = B[1]
                    branch[y, x + dx] = True
    # the lobes: a big one behind, the rest round its top and sides; lower ones in front
    lobes_ = [(ccx, ccy, min(rx, ry) * 0.78)]
    for i in range(lobes - 1):
        a = -math.pi / 2 + (i - (lobes - 2) / 2) * (math.pi * 1.25 / max(lobes - 2, 1)) + rng.uniform(-0.25, 0.25)
        r = min(rx, ry) * rng.uniform(0.48, 0.62)
        d = rng.uniform(0.5, 0.72)                      # out from the middle: an uneven outline
        lobes_.append((ccx + math.cos(a) * rx * d * stretch, ccy + math.sin(a) * ry * d, r))
    # the lowest lobes hang either side of the trunk, leaving a notch where it goes up into the crown
    for side in (-1, 1):
        lobes_.append((ccx + side * rx * rng.uniform(0.38, 0.5), ccy + ry * rng.uniform(0.3, 0.42), min(rx, ry) * rng.uniform(0.46, 0.56)))
    lobes_.sort(key=lambda l: l[1])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    owner = np.full((h, w), -1, dtype=np.int16)
    light1 = np.zeros((h, w))
    for i, (lx, ly, r) in enumerate(lobes_):
        dx, dy = xx - lx, (yy - ly) / 0.9
        d, ang = np.hypot(dx, dy), np.arctan2(dy, dx)
        n = max(5, round(2 * math.pi * r / 4.0))
        bump = 1.3 * np.abs(np.sin(n * ang / 2 + rng.uniform(0, 6.3)))     # a scalloped, leafy rim
        inside = d < r - 0.8 + bump
        owner[inside] = i
        light1[inside] = (-(dx * 0.62 + dy * 0.78) / r)[inside]
    crown_m = owner >= 0
    # the whole crown as one ball, darkest underneath
    light2 = -((xx - ccx) * 0.62 + (yy - ccy) * 0.78) / max(rx, ry)
    under = np.clip((yy - ccy) / (ry * 0.95), 0, 1) ** 1.3
    # a lobe in front throws a shadow on the lobe behind it, along its own upper-left edge
    shade = np.zeros((h, w), dtype=bool)
    for ox, oy in ((1, 2), (2, 2), (0, 2), (2, 1)):
        ahead = np.full((h, w), -1, dtype=np.int16)
        ahead[:h - oy, :w - ox] = owner[oy:, ox:]
        shade |= crown_m & (ahead > owner)
    # leaf texture: small staggered cells, felt only where the light is near a cut between tones
    cell = _hash(np.floor((xx + (np.floor(yy / 2) % 2) * 1.5) / 3), np.floor(yy / 2), seed) - 0.5
    v = 0.6 * light1 + 0.4 * light2 - 0.8 * shade - 0.6 * under + 0.42 * cell
    tone = np.digitize(v, [-0.8, -0.32, 0.12, 0.55])                    # 0 deepest .. 4 lightest
    crown_idx = np.where(crown_m, np.array(L)[tone], T)
    # a gap or two low in the crown: shade inside, a branch across it
    for g in range(gaps):
        gx = ccx + rng.uniform(-0.45, 0.45) * rx
        gy = ccy + ry * rng.uniform(0.25, 0.5)
        hole = (np.hypot(xx - gx, (yy - gy) * 1.3) < rng.uniform(1.6, 2.4)) & crown_m
        crown_idx[hole] = L[0]
        bx = int(round(gx)) - 1
        for k in range(4):
            y, x = int(round(gy)) + 1 - k // 2, bx + k
            if 0 <= y < h and 0 <= x < w and hole[y, x]:
                crown_idx[y, x] = B[1]
    idx = np.where(crown_m, crown_idx, idx)
    # a branch shows only where the crown is round it - in the notch - never sticking out into the air
    near_crown = crown_m.copy()
    for oy, ox in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        near_crown |= _shift(crown_m, oy, ox)
    idx[branch & ~crown_m & ~near_crown] = T
    # the crown's shadow on the trunk just below it
    for x in range(w):
        col = np.nonzero(crown_m[:, x])[0]
        if len(col):
            for y in range(col.max() + 1, min(col.max() + 4, h)):
                if idx[y, x] in B[2:]:
                    idx[y, x] = B[1]
    idx = _clean(idx, keep=B)
    # the rim: the darkest green, a step lighter where the light falls on it; the trunk in ink
    filled = idx != T
    edge = np.zeros_like(filled)
    for oy, ox in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        edge |= _shift(filled, oy, ox)
    edge &= ~filled
    lit_side = _shift(np.isin(idx, L[3:]), -1, -1)   # the lit crown just below-right of it
    crown_near = np.zeros_like(filled)
    for oy, ox in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        crown_near |= _shift(np.isin(idx, L), oy, ox)
    idx = np.where(edge & crown_near, np.where(lit_side, L[1], L[0]), idx)
    idx = np.where(edge & ~crown_near, B[0], idx)
    return _image(idx, names, P)


def conifer(P, w=30, h=58, tiers=7, seed=0):
    """A pine: tiers of drooping branches from the top down, ragged at the tips, lit on the left,
    each laying a band of shadow on the tier below; a short trunk under the lowest."""
    rng = random.Random(f"pine:{seed}")
    names = LEAF["pine"] + BARK["pine"]
    L = [1, 2, 3, 4, 5]
    B = [6, 7, 8, 9]
    idx = np.zeros((h, w), dtype=np.int16)
    cx = w / 2 - 0.5
    _trunk(idx, cx, h - 12, h - 2, 4, B, rng)
    top, bottom = 1, h - 9
    step = (bottom - top) / tiers
    tier_of = np.full((h, w), -1, dtype=np.int16)
    for i in range(tiers):
        y0 = top + i * step * 0.9
        th = step * 1.5
        half_w = 1.5 + (w / 2 - 1.5) * ((i + 1) / tiers) ** 0.85
        tips = {xx: rng.choice([0, 0, 1, 2]) for xx in range(w)}
        for yy in range(int(y0), min(int(y0 + th) + 2, h)):
            t = (yy - y0) / th
            hw = half_w * min(1.0, max(t, 0.0) * 1.5) ** 0.85
            for xx in range(int(cx - hw), int(cx + hw) + 2):
                if not 0 <= xx < w:
                    continue
                u = (xx - cx) / max(hw, 1)
                # the skirt hangs lower at its ends, ragged where each branch tip ends
                edge_t = 0.66 + 0.34 * (1 - abs(u) ** 2) + tips[xx] / th
                if t > edge_t:
                    continue
                lit = -u * 0.85 - (t - 0.35) * 0.7 + 0.25 * (_hash(xx // 2, yy, i) - 0.5)
                tone = 4 if lit > 0.7 else 3 if lit > 0.2 else 2 if lit > -0.35 else 1
                idx[yy, xx] = L[tone]
                tier_of[yy, xx] = i
    # each tier's shadow on the one below: the first rows of a tier just under the one above
    for yy in range(1, h):
        for xx in range(w):
            if tier_of[yy, xx] > 0 and tier_of[yy - 1, xx] >= 0 and tier_of[yy - 1, xx] < tier_of[yy, xx]:
                for k in range(2):
                    if yy + k < h and tier_of[yy + k, xx] == tier_of[yy, xx]:
                        idx[yy + k, xx] = L[0] if k == 0 else max(L[1], idx[yy + k, xx] - 1)
    idx = _clean(idx, keep=B)
    filled = idx != T
    edge = np.zeros_like(filled)
    for oy, ox in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        edge |= _shift(filled, oy, ox)
    edge &= ~filled
    near_leaf = np.zeros_like(filled)
    for oy, ox in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        near_leaf |= _shift(np.isin(idx, L), oy, ox)
    idx = np.where(edge, np.where(near_leaf, L[0], B[0]), idx)
    return _image(idx, names, P)


def oak(P, seed=0):
    return broadleaf(P, "oak", 44, 56, lobes=6, crown=(0.5, 0.35, 0.45, 0.33), trunk_w=6, seed=seed, gaps=0)


def alder(P, seed=0):
    return broadleaf(P, "alder", 32, 52, lobes=5, crown=(0.5, 0.36, 0.42, 0.36), trunk_w=4, seed=seed, gaps=0)


def birch(P, seed=0):
    return broadleaf(P, "birch", 28, 52, lobes=5, crown=(0.5, 0.3, 0.42, 0.27), trunk_w=3, seed=seed, gaps=0)


def windbent(P, seed=0):
    """A tree the west wind shapes: low, the trunk leaning east, the crown pushed east and flat."""
    return broadleaf(P, "oak", 40, 34, lobes=5, crown=(0.58, 0.34, 0.36, 0.28), trunk_w=4, seed=seed, lean=-0.35,
                     stretch=1.35, gaps=0)


def pine(P, seed=0):
    return conifer(P, 30, 58, 7, seed)
