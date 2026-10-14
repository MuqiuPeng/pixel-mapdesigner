"""Draws a map so that it does not look like a grid - and so that it moves.

The map is still cells underneath (the simulation needs them), but nothing here stamps one
picture per cell:

- **Ground** is a material per cell, drawn in world-pixel space. Borders between materials are
  bent by noise and rounded at the corners; texture comes from noise at world coordinates, so
  it never repeats. Higher-ranked materials lie on top of lower ones, with a rim, and may drop
  a cliff or bank on their south side. The lower one can answer with shore bands.
- **Decor** (tufts, flowers, pebbles) is sprinkled by each material at pixel positions.
- **Objects** are sprites of any size standing on a cell: sorted by their feet so tall ones
  overlap what is behind, nudged and mirrored per cell, with a shadow on the ground.

**Time.** A picture is one frame of a loop of FRAMES frames at FPS (four seconds). What moves
is declared, not painted frame by frame: a speck of a material may `twinkle` or `drift`
(ripples), a shore may `lap`, an object has a `motion` (`sway`, `flicker`, `bob`) that is
derived from its one sprite. Each thing starts at its own phase, so a wood does not sway in
step. Small motions repeat every CYCLE frames; two things take the whole loop:

- **Wind.** Gusts cross the map as ragged bands, one after another. Where a gust is, a
  ground's `gust` specks show (grass laid flat shows its pale side) and anything whose
  motion is `gust` leans. Between gusts the grass is still.
- **Current.** Running water (`flows`) moves toward what it `joins`: a river's ripples
  travel down it to the sea, and go on a little way out from the mouth. Still water does not.
- **Water's own motion.** Each kind of water moves the way it does (`water`): the sea's swell
  comes in as long crests parallel to the shore and meets the surf; wind ruffles open water
  in the same gusts that lay the grass flat; a bog pool sends up bubbles and a tide pool takes a
  drip, each a ring that widens and is gone. A river's current is quickest mid-stream: its
  ripples are thicker and run faster there, few and slow along the banks.
- **Surf.** A ground with `surf` (the sea) sends waves at the land: a line of foam comes in,
  runs up the beach, slides back, and leaves the sand dark and wet until it dries. Each
  stretch of shore is at its own point in that, so the coast is never all in step.

**Height.** A map that knows how high each cell stands (`relief`) is drawn in steps, the way
Stardew draws its hills. Only where the land is steep is a step a cliff (mapkit.cliffs - the same
rule the checks and the game walk by); on gentle ground a step is a slope and nothing is drawn.
Where a cliff stands above the ground to its south there is a face, as tall as the step; where
the map has a ramp, the face is cut into stairs. Each ground says what its face is made of
(`face`) and how its lip catches the light (`lip`): turf shows a bank of earth, rock shows rock,
running water falls white over the step. On the east and west sides a step is only a dark line.

Everything is a pure function of world position and frame, so any region renders the same
alone as it does inside the whole map, and the editor can redraw just the part that changed.

Colours are palette indices until the very end: pixels stay hard and inside the palette.
"""
from __future__ import annotations

import math
import re
import zlib

import numpy as np
from PIL import Image, ImageFilter

import mapkit
from mapkit import CELL as C, MapError, SHAPES

PAD = 4          # cells of context drawn around a region and cropped away
MAX_SPRITE = 64  # so PAD cells are enough for the tallest object behind the region
FRAMES = 24      # length of the whole loop: gusts and waves take all of it
CYCLE = 8        # small motions (sway, ripples, flicker) repeat this often; divides FRAMES
FPS = 6
MOTIONS = ("none", "sway", "flicker", "bob", "gust")
SPECK_ANIMS = ("twinkle", "drift", "gust", "flow")
WATER_MOVES = ("surface", "swell", "ruffle", "rings")
SWELL_REACH = 4  # cells: how far out from the land the swell is drawn - no more than PAD, so
                 # every pixel's crest is decided by land inside its window
WIND = (1.0, 0.35)   # the way gusts travel
GUST_SPAN = 230.0    # px from one gust to the next along the wind
GUST_WIDTH = 0.2     # how much of that a gust covers
# How firmly borders keep to the grid: 0 lets them wander freely, 1 lays them along cell edges
# with only the corners rounded. The map is a grid underneath and should still read as one -
# above all where it matters to the feet: a step in the land runs along the edge between two
# cells, so it is plain which cells it stands between.
SNAP_GROUND = 0.5
SNAP_STEP = 0.85
TIER = 8          # relief units to a step (some two men high, at the usual scale)
FACE = 6          # pixels of face for each step down
# The one light over everything. Sprites are drawn lit from the top left, so nothing that is
# shaded may be mirrored; and every shadow on the ground falls the same way, away from the sun:
# `x`, `y` how far it reaches per px of the thing's height. The time of day is this and only
# this - morning, noon and evening move and stretch every shadow together.
SUN = {"x": 0.32, "y": 0.1}
_SWAY = [0, 0, 1, 1, 0, 0, -1, -1]
_LAP = [0, 0, 0, 1, 1, 1, 1, 0]


def _seed(text) -> int:
    return zlib.crc32(str(text).encode("utf-8")) & 0xFFFF


_XY = [None, None, None, None]   # the window's pixel grids, and the same as uint32: hashes of them are kept while it is drawn
_HASHED = {}


def _hash(ix, iy, seed: int):
    """0..1 from integer coordinates; works on numbers and on arrays alike. Done in 32-bit
    unsigned arithmetic, which wraps exactly as the masks it replaces did; over the window's
    whole pixel grid the result is kept, since many looks ask for the same one."""
    whole = ix is _XY[0] and iy is _XY[1] and ix is not None
    if whole and seed in _HASHED:
        return _HASHED[seed]
    ux = _XY[2] if whole else np.asarray(ix).astype(np.uint32)
    uy = _XY[3] if whole else np.asarray(iy).astype(np.uint32)
    with np.errstate(over="ignore"):
        h = (ux * np.uint32(374761393) + uy * np.uint32(668265263)
             + np.uint32((seed * 974711) & 0xFFFFFFFF))
        h = (h ^ (h >> np.uint32(13))) * np.uint32(1274126177)
    # 24 bits over 2^24: exact in float32, at half the memory float64 would take
    out = ((h ^ (h >> np.uint32(16))) & np.uint32(0xFFFFFF)).astype(np.float32) * np.float32(1 / 16777216)
    if np.ndim(out) == 0:
        return float(out)          # a number in, a plain number out (it may end up in JSON)
    if whole:
        _HASHED[seed] = out
    return out


def _noise(X, Y, wavelength: float, seed: int):
    """Smooth value noise, 0..1, in world pixels. X and Y are a meshgrid, so the lattice is
    hashed once per lattice point rather than once per pixel."""
    fx, fy = X[0] / wavelength, Y[:, 0] / wavelength
    ix, iy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    tx, ty = fx - ix, fy - iy
    tx, ty = (tx * tx * (3 - 2 * tx))[None, :], (ty * ty * (3 - 2 * ty))[:, None]
    lat = _hash(np.arange(ix[0], ix[-1] + 2)[None, :], np.arange(iy[0], iy[-1] + 2)[:, None], seed).astype(np.float32)
    jx, jy = ix - ix[0], iy - iy[0]
    # across first, on the lattice's own few rows; then down, once per pixel row
    rows = lat[:, jx] * (1 - tx) + lat[:, jx + 1] * tx
    return rows[jy] * (1 - ty) + rows[jy + 1] * ty


def _shift(a, dx: int, dy: int, fill=0):
    """`a` moved by (dx, dy) without wrapping."""
    out = np.full_like(a, fill)
    h, w = a.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = a[ys, xs]
    return out


class Palette:
    """Palette names as small integers, and the step to the next darker colour of a ramp
    (g3 -> g2 -> g1 -> g0), which is how shadows stay inside the palette."""

    def __init__(self, pal: dict):
        self.names = [n for n in pal if not n.startswith("_")]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.missing = len(self.names)
        self.rgba = np.array([pal[n] for n in self.names] + [mapkit.MISSING], dtype=np.uint8)
        dark = list(range(len(self.names) + 1))
        for name, i in self.index.items():
            m = re.match(r"^([a-z]+)(\d)$", name)
            if m and f"{m[1]}{int(m[2]) - 1}" in self.index:
                dark[i] = self.index[f"{m[1]}{int(m[2]) - 1}"]
        self.dark = np.array(dark, dtype=np.int16)

    def __getitem__(self, name) -> int:
        if name not in self.index:
            raise MapError(f"调色板里没有颜色 {name!r}")
        return self.index[name]

    def sprite(self, art: dict):
        """Art as an array of palette indices, -1 where transparent."""
        rows, key = art["rows"], art["key"]
        out = np.full((len(rows), len(rows[0])), -1, dtype=np.int16)
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch in key:
                    out[y, x] = self[key[ch]]
        return out

    def image(self, idx) -> Image.Image:
        return Image.fromarray(self.rgba[np.where(idx < 0, 0, idx)] * (idx >= 0)[..., None].astype(np.uint8))


def moved(sprite, motion: str, frame: int):
    """One sprite, one frame of its motion. Only what moves is moved: a tree's crown leans and
    the trunk stays; a flame's tip wavers; a body dips by a pixel."""
    if not motion or motion == "none":
        return sprite
    h = sprite.shape[0]
    out = sprite.copy()

    def lean(y0, y1, dx):
        if dx == 0 or y0 >= y1:
            return
        rows = sprite[y0:y1]
        new = np.full_like(rows, -1)
        if dx > 0:
            new[:, dx:] = rows[:, :-dx]
        else:
            new[:, :dx] = rows[:, -dx:]
        out[y0:y1] = new

    if motion == "sway":  # the top leads, the middle follows a frame later
        a, b = max(1, round(h * 0.3)), max(1, round(h * 0.55))
        lean(0, a, _SWAY[frame % CYCLE])
        lean(a, b, _SWAY[(frame - 1) % CYCLE])
    elif motion == "gust":  # frame is 0 (standing) or 1 (pushed over by the wind)
        if frame:
            lean(0, max(1, round(h * 0.5)), 1)
    elif motion == "flicker":
        a, b = max(1, round(h * 0.3)), max(1, round(h * 0.55))
        lean(0, a, [0, 1, 0, -1][frame % 4])
        lean(a, b, [0, 0, 1, 0][frame % 4])
    elif motion == "bob" and frame % CYCLE >= CYCLE // 2 and h > 4:
        out[1:h - 2] = sprite[0:h - 3]
        out[0] = -1
    return out


# ---------------------------------------------------------------- what a material looks like

def clean_look(look, pal: dict, tiles: dict) -> dict:
    """Check a ground tile's `look` and return it in canonical form.

        base     colour of the surface
        patches  [{color, scale, amount, fill?}]  soft areas of another tone; scale in px, amount
                 0..1; fill below 1 breaks the area into scattered pixels, so it reads as texture
                 and not as a border between two grounds
        specks   [{color, shape, density, anim?}]  small marks; shape from mapkit.SHAPES, density
                 per px; anim "twinkle" (comes and goes), "drift" (slides along, like a ripple)
                 "gust" (shows only while a gust of wind is passing) or "flow" (travels with the
                 current; on ground that does not itself flow it shows only where a current
                 reaches, as off a river's mouth)
        rim      colour of the edge where this lies on a lower material (or null)
        face     [colours top .. bottom]  what a step of the land shows where this ground is on
                 top of it (default: two shades darker than base)
        lip      colour of the edge at the top of such a step (default: base)
        edge     [{color, width, wobble?, density?, shape?}]  the way this ground changes just
                 inside its border with a lower one: width in px, wobble makes the width uneven,
                 density below 1 scatters it (thinning away from the border) instead of filling
        apron    the same, but laid on the lower ground just outside the border: what this
                 ground spills onto its neighbour (grass thinning into sand, scree below rock);
                 a layer may say `over: [tile ids]` to spill only onto those
        cliff    {colors: [top .. bottom], over?: [tile ids], cracks?: bool}
                 a face dropped on the south side, one colour per pixel row
        shore    {colors: [nearest .. farthest], lap?: bool}  bands inside this material along
                 higher ones; with lap the nearest two trade places, like water at an edge
        surf     {foam, water, reach, onto?: [tile ids]}  waves: foam comes in over this ground,
                 runs `reach` px up the grounds in `onto` (default: any higher) as `water` with
                 a `foam` front, slides back and leaves them wet
        water    [{move, color, glint?, spacing?, density?}]  water's own motion: move "swell"
                 (crests every `spacing` px coming in toward the land), "ruffle" (while a gust
                 passes, the surface breaks into ripples; `glint` catches the light) or "rings"
                 (bubbles or drips: a ring widens from a point and is gone; density per px)
        flows    true for running water: it has a current, toward the ground it `joins` (or, if
                 it reaches none, off the bottom or side of the map; a pond has none)
        joins    [tile ids]  grounds this one runs into without a border: where a river meets
                 the sea there is no shore, no rim, no surf - none of what either does at
                 land. It is enough for one of the two to say so.
        decor    [{tile, density}]  object tiles sprinkled on it; density per cell, 0..1
    """
    if not isinstance(look, dict):
        raise MapError("地面图块需要 look（外观）")

    def color(name):
        if not isinstance(name, str) or name.startswith("_") or name not in pal:
            raise MapError(f"调色板里没有颜色 {name!r}")
        return name

    def number(v, lo, hi, what):
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise MapError(f"{what} 应该是数字")
        if not lo <= v <= hi:
            raise MapError(f"{what} 要在 {lo} 到 {hi} 之间")
        return v

    def items(key, limit):
        v = look.get(key) or []
        if not isinstance(v, list) or len(v) > limit:
            raise MapError(f"{key} 应该是最多 {limit} 项的列表")
        return v

    out = {"base": color(look.get("base")), "patches": [], "specks": [], "rim": None, "edge": [], "apron": [],
           "cliff": None, "shore": None, "surf": None, "flows": bool(look.get("flows")), "joins": [], "decor": [],
           "water": []}
    try:
        for p in items("patches", 4):
            out["patches"].append({"color": color(p["color"]), "scale": number(p.get("scale", 24), 4, 96, "patch.scale"),
                                   "amount": number(p.get("amount", 0.3), 0, 1, "patch.amount"),
                                   "fill": number(p.get("fill", 1), 0.05, 1, "patch.fill")})
        for s in items("specks", 6):
            shape, anim = s.get("shape", "dot"), s.get("anim") or None
            if shape not in SHAPES:
                raise MapError(f"没有形状 {shape!r}，可用：{', '.join(SHAPES)}")
            if anim not in (None, *SPECK_ANIMS):
                raise MapError(f"speck.anim 只能是 {' / '.join(SPECK_ANIMS)} 或不填")
            out["specks"].append({"color": color(s["color"]), "shape": shape,
                                  "density": number(s.get("density", 0.01), 0, 0.3, "speck.density"), "anim": anim})
        if look.get("rim"):
            out["rim"] = color(look["rim"])
        if look.get("face"):
            if not isinstance(look["face"], list) or not 1 <= len(look["face"]) <= 8:
                raise MapError("face 应该是 1 到 8 个颜色")
            out["face"] = [color(c) for c in look["face"]]
        if look.get("lip"):
            out["lip"] = color(look["lip"])
        for key in ("edge", "apron"):
            for z in items(key, 5):
                shape = z.get("shape", "dot")
                if shape not in SHAPES:
                    raise MapError(f"没有形状 {shape!r}，可用：{', '.join(SHAPES)}")
                over = list(z.get("over") or []) if key == "apron" else []
                for t in over:
                    if tiles.get(t, {}).get("layer") != "ground":
                        raise MapError(f"apron.over 里的 {t!r} 不是地面图块")
                out[key].append({"color": color(z["color"]), "width": number(z.get("width", 3), 1, 16, f"{key}.width"),
                                 "wobble": number(z.get("wobble", 0), 0, 8, f"{key}.wobble"),
                                 "density": number(z.get("density", 1), 0.01, 1, f"{key}.density"), "shape": shape,
                                 **({"over": over} if key == "apron" else {})})
        if look.get("cliff") and look["cliff"].get("colors"):
            cl = look["cliff"]
            if len(cl["colors"]) > 16:
                raise MapError("cliff 最多 16 行")
            over = [t for t in cl.get("over") or []]
            for t in over:
                if tiles.get(t, {}).get("layer") != "ground":
                    raise MapError(f"cliff.over 里的 {t!r} 不是地面图块")
            out["cliff"] = {"colors": [color(c) for c in cl["colors"]], "over": over, "cracks": bool(cl.get("cracks"))}
        if look.get("shore") and look["shore"].get("colors"):
            if len(look["shore"]["colors"]) > 8:
                raise MapError("shore 最多 8 条")
            out["shore"] = {"colors": [color(c) for c in look["shore"]["colors"]], "lap": bool(look["shore"].get("lap"))}
        if look.get("surf"):
            sf = look["surf"]
            onto = list(sf.get("onto") or [])
            for t in onto:
                if tiles.get(t, {}).get("layer") != "ground":
                    raise MapError(f"surf.onto 里的 {t!r} 不是地面图块")
            out["surf"] = {"foam": color(sf.get("foam")), "water": color(sf.get("water")),
                           "reach": int(number(sf.get("reach", 7), 2, 14, "surf.reach")), "onto": onto}
        for wv in items("water", 4):
            if wv.get("move") not in WATER_MOVES:
                raise MapError(f"water.move 只能是 {' / '.join(WATER_MOVES)}")
            out["water"].append({"move": wv["move"], "color": color(wv.get("color")),
                                 "glint": color(wv["glint"]) if wv.get("glint") else None,
                                 "spacing": int(number(wv.get("spacing", 12), 6, 24, "water.spacing")),
                                 "density": number(wv.get("density", 0.002), 0, 0.05, "water.density")})
        for t in items("joins", 8):
            if tiles.get(t, {}).get("layer") != "ground":
                raise MapError(f"joins 里的 {t!r} 不是地面图块")
            out["joins"].append(t)
        for d in items("decor", 8):
            if tiles.get(d["tile"], {}).get("layer") != "object":
                raise MapError(f"decor 里的 {d.get('tile')!r} 不是物件图块")
            out["decor"].append({"tile": d["tile"], "density": number(d.get("density", 0.05), 0, 1, "decor.density")})
    except (KeyError, TypeError, AttributeError) as e:
        raise MapError(f"look 的写法有误：{e!r}")
    return out


# ---------------------------------------------------------------- the ground

def currents(m, tiles: dict):
    """Which way the water runs, per cell: two arrays (x and y parts, length 0..1), or None if
    nothing on the map flows. Worked out on the whole map, because where a river is going
    cannot be told from a piece of it.

    Each stretch of running water flows toward the cells where it meets a ground it joins
    (the sea); failing that, if the map has heights, toward the lowest place where it meets
    the map's edge; failing that, toward where it leaves at the bottom, right or left.
    Off a mouth the current carries on a few cells into the ground it joined, weakening."""
    W, H, cells = m.width, m.height, m.layers["ground"]

    def look(x, y):
        t = tiles.get(cells[y][x]) if cells[y][x] else None
        return t["look"] if t and t["layer"] == "ground" else None

    runs = [[bool(look(x, y) and look(x, y).get("flows")) for x in range(W)] for y in range(H)]
    if not any(map(any, runs)):
        return None
    kin = {}
    for t in tiles.values():
        if t["layer"] == "ground":
            for other in t["look"].get("joins", []):
                kin.setdefault(t["id"], set()).add(other)
                kin.setdefault(other, set()).add(t["id"])
    near4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    near8 = near4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))
    dist = [[-1] * W for _ in range(H)]
    vx, vy = np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)
    seen, mouths = [[False] * W for _ in range(H)], []
    for sy in range(H):
        for sx in range(W):
            if not runs[sy][sx] or seen[sy][sx]:
                continue
            body, todo = [], [(sx, sy)]
            seen[sy][sx] = True
            while todo:
                x, y = todo.pop()
                body.append((x, y))
                for dx, dy in near4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and runs[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        todo.append((nx, ny))
            out = {}  # cell where it leaves -> the way out
            for x, y in body:
                for dx, dy in near4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and not runs[ny][nx] and cells[ny][nx] in kin.get(cells[y][x], ()):
                        ox, oy = out.get((x, y), (0, 0))
                        out[(x, y)] = (ox + dx, oy + dy)
                        mouths.append((nx, ny, x, y))
            relief = getattr(m, "relief", None)
            if not out and relief and max(relief[y][x] for x, y in body) - min(relief[y][x] for x, y in body) >= 2:
                # It reaches no sea here, but the map knows how the land lies: it runs downhill,
                # each cell toward the lower water around it.
                here = set(body)
                for x, y in body:
                    ax = ay = 0.0
                    for dx, dy in near8:
                        if (x + dx, y + dy) in here:
                            fall = (relief[y][x] - relief[y + dy][x + dx]) / (dx * dx + dy * dy)
                            ax, ay = ax + fall * dx, ay + fall * dy
                    n = (ax * ax + ay * ay) ** 0.5
                    if n:
                        vx[y, x], vy[y, x] = ax / n, ay / n
                    dist[y][x] = 0
                continue
            if not out:
                for edge, way in ((lambda c: c[1] == H - 1, (0, 1)), (lambda c: c[0] == W - 1, (1, 0)), (lambda c: c[0] == 0, (-1, 0))):
                    out = {c: way for c in body if edge(c)}
                    if out:
                        break
            if not out:
                continue  # a pond
            todo = list(out)
            for x, y in todo:
                dist[y][x] = 0
            for x, y in todo:  # breadth first: the list grows as it is walked
                for dx, dy in near4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and runs[ny][nx] and dist[ny][nx] < 0:
                        dist[ny][nx] = dist[y][x] + 1
                        todo.append((nx, ny))
            for x, y in body:
                ax, ay = out.get((x, y), (0, 0))
                ax, ay = ax * 2.0, ay * 2.0
                for dx, dy in near8:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and dist[ny][nx] >= 0:
                        pull = (dist[y][x] - dist[ny][nx]) / (dx * dx + dy * dy)
                        ax, ay = ax + pull * dx, ay + pull * dy
                n = (ax * ax + ay * ay) ** 0.5
                if n:
                    vx[y, x], vy[y, x] = ax / n, ay / n
    # even it out along the stream, so it bends rather than turns corners
    sx, sy = vx.copy(), vy.copy()
    for y in range(H):
        for x in range(W):
            if dist[y][x] < 0:
                continue
            ax, ay = float(vx[y, x]) * 2, float(vy[y, x]) * 2
            for dx, dy in near8:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and dist[ny][nx] >= 0:
                    ax, ay = ax + float(vx[ny, nx]), ay + float(vy[ny, nx])
            n = (ax * ax + ay * ay) ** 0.5
            if n:
                sx[y, x], sy[y, x] = ax / n, ay / n
    # and out past the mouth, into the water it joined
    reach = {}
    front = []
    for nx, ny, x, y in mouths:
        if (nx, ny) not in reach:
            reach[(nx, ny)] = (float(sx[y, x]), float(sy[y, x]), 0)
            front.append((nx, ny))
    for x, y in front:
        ax, ay, depth = reach[(x, y)]
        if depth >= 3:
            continue
        for dx, dy in near4:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and not runs[ny][nx] and (nx, ny) not in reach \
                    and cells[ny][nx] == cells[y][x] and dx * ax + dy * ay > -0.3:
                reach[(nx, ny)] = (ax, ay, depth + 1)
                front.append((nx, ny))
    for (x, y), (ax, ay, depth) in reach.items():
        k = (0.85, 0.6, 0.4, 0.22)[depth]
        sx[y, x], sy[y, x] = ax * k, ay * k
    return sx, sy


def _grow(mask):
    return mask | _shift(mask, 1, 0) | _shift(mask, -1, 0) | _shift(mask, 0, 1) | _shift(mask, 0, -1)


def _reach(src, into, n: int):
    """How many pixels each pixel of `into` is from `src` (1..n), going only through `into`;
    0 where it is farther, or not in `into`."""
    d = np.zeros(src.shape, dtype=np.int8)
    reach = src
    for i in range(1, n + 1):
        ring = _grow(reach) & ~reach & into
        if not ring.any():
            break
        d[ring] = i
        reach = reach | ring
    return d


def _zone(idx, d, layers, X, Y, seed: int, P: Palette, where, only=None):
    """Paint a border zone: each layer a band of its width (unevenly wide if it wobbles),
    solid, or scattered and thinning with distance. `only(layer)` may narrow where a layer
    is allowed. Returns the pixels it touched."""
    touched = np.zeros(d.shape, dtype=bool)
    everywhere = where
    for i, z in enumerate(layers):
        where = everywhere & only(z) if only else everywhere
        limit = z["width"] + z["wobble"] * (2 * _noise(X, Y, 9, seed + 17 * (i + 1)) - 1)
        sel = (d > 0) & (d <= limit) & where
        if z["density"] < 1:
            thin = 1 - (d - 1) / (z["width"] + z["wobble"])
            sel &= _hash(X, Y, seed + 59 * (i + 1)) < z["density"] * thin
            for ox, oy in SHAPES[z["shape"]]:
                part = _shift(sel, ox, oy) & where
                idx[part] = P[z["color"]]
                touched |= part
        else:
            idx[sel] = P[z["color"]]
            touched |= sel
    return touched


def _ground(G, X, Y, grounds, P: Palette, VX=None, VY=None, RF=None, CL=None, PL=None) -> dict:
    """Everything about the ground that does not depend on the frame: which material owns
    each pixel, the still picture, and what is left to animate."""
    h, w = X.shape
    present = [int(k) for k in np.unique(G)]
    big = np.repeat(np.repeat(G, C, 0), C, 1)
    if len(present) == 1:
        M = big.copy()
    else:
        # Each material claims a pixel by how much of it is nearby, bent by its own noise;
        # the strongest claim wins. That rounds corners and makes every border wander.
        # The noise fades to nothing at the middle of each cell, so however far a border
        # strays, the centre of a cell always shows the material the cell really has.
        u, v = np.abs((X % C) - (C - 1) / 2) / (C / 2), np.abs((Y % C) - (C - 1) / 2) / (C / 2)
        sway = np.minimum(1.0, 1.6 * np.maximum(u, v)).astype(np.float32)
        best = np.full((h, w), -9.0, dtype=np.float32)
        M = np.zeros((h, w), dtype=np.int16)
        for k in present:
            near = Image.fromarray(((big == k) * 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(5.5 - 2.5 * SNAP_GROUND))
            s = _seed(grounds[k - 1]["id"]) if 0 < k <= len(grounds) else 1
            f = (np.asarray(near, dtype=np.float32) / 255 + k * 1e-4
                 + sway * (1 - SNAP_GROUND) * (0.9 * (_noise(X, Y, 12, s) - 0.5) + 0.64 * (_noise(X, Y, 31, s + 7) - 0.5)))
            win = f > best
            best[win] = f[win]
            M[win] = k

    idx = np.zeros((h, w), dtype=np.int16)
    void, unknown = M == 0, M > len(grounds)
    idx[void] = P["n0"]
    idx[unknown] = P.missing
    rank = np.where(unknown, 0, M)  # the order of `grounds` is the order of ranks
    fixed = np.zeros((h, w), dtype=bool)  # rims, still shores and cliffs: nothing animated may cross them
    specks, laps, gusts, surfs, streams, waters = [], [], [], [], [], []
    # Grounds that run into each other (a river into the sea) are not each other's shore:
    # for every border effect below, a ground's kin count as neither higher nor lower.
    code = {t["id"]: i + 1 for i, t in enumerate(grounds)}
    joined = {k: set() for k in code.values()}
    for t in grounds:
        for other in t["look"].get("joins", []):
            if other in code:
                joined[code[t["id"]]].add(code[other])
                joined[code[other]].add(code[t["id"]])
    kin = {k: np.isin(M, list(v)) if v else np.zeros((h, w), dtype=bool) for k, v in joined.items()}
    # On a map with heights, a step in the land is drawn where the land really steps (below);
    # a material's own rim and face would wall it in against every lower material, height or
    # no height. There they are kept only against still water - rock standing up out of the
    # sea, which the steps do not draw (water lies level).
    still_water = None
    if RF is not None:
        still_water = np.isin(M, [i + 1 for i, t in enumerate(grounds) if t["pass"] == "swim" and not t["look"].get("flows")])

    for k in present:
        if not 0 < k <= len(grounds):
            continue
        t = grounds[k - 1]
        look, s, mine = t["look"], _seed(t["id"]), M == k
        idx[mine] = P[look["base"]]
        for i, p in enumerate(look["patches"]):
            area = mine & (_noise(X, Y, p["scale"], s + 31 * (i + 1)) > 1 - p["amount"] * 0.75)
            if p.get("fill", 1) < 1:
                area &= _hash(X, Y, s + 43 * (i + 1)) < p["fill"]
            idx[area] = P[p["color"]]
        bank = None   # running water: how far from the bank (0..1 for 0..8 px), where the current is quicker
        if look.get("flows") and any(sp.get("anim") == "flow" for sp in look["specks"]):
            b_ = _reach(~mine & ~kin[k], mine, 8)
            bank = np.where(b_ == 0, 8, b_).astype(np.float32) / 8
        for i, sp in enumerate(look["specks"]):
            chance = _hash(X, Y, s + 101 * (i + 1)).astype(np.float32)
            seeds = (chance < sp["density"]) & mine
            if sp.get("anim") == "gust":
                gusts.append((mine, sp, chance))
                continue
            if sp.get("anim") == "flow":
                if look.get("flows"):
                    # running water: everywhere (it only drifts where it is still), but the
                    # current is quickest mid-stream - more ripples there, few by the banks
                    sel = seeds & (chance < sp["density"] * (0.3 + 0.7 * bank))
                elif VX is None:
                    continue             # other grounds: only where a current reaches them
                else:
                    strength = np.hypot(VX, VY)
                    sel = (chance < sp["density"] * strength) & mine & (strength > 0.15)
                ys, xs = np.nonzero(sel)
                streams.append({"mine": mine, "sp": sp, "ys": ys, "xs": xs,
                                "fast": (0.55 + 0.75 * bank[ys, xs]) if look.get("flows") else np.ones(len(ys), dtype=np.float32),
                                "ph": np.floor(_hash(X[ys, xs], Y[ys, xs], s + 211 * (i + 1)) * CYCLE).astype(np.int64),
                                "vx": VX[ys, xs] if VX is not None else np.zeros(len(ys), dtype=np.float32),
                                "vy": VY[ys, xs] if VY is not None else np.zeros(len(ys), dtype=np.float32)})
                continue
            if sp.get("anim"):
                phase = np.floor(_hash(X, Y, s + 211 * (i + 1)) * CYCLE).astype(np.int8)
                specks.append((mine, sp, seeds, phase))
                continue
            for ox, oy in SHAPES[sp["shape"]]:
                idx[_shift(seeds, ox, oy) & mine] = P[sp["color"]]
        if look.get("edge"):
            width = int(max(z["width"] + z["wobble"] for z in look["edge"])) + 1
            fixed |= _zone(idx, _reach((rank < k) & ~mine & ~kin[k], mine, width), look["edge"], X, Y, s + 300, P, mine)
        if look.get("surf"):
            sf = look["surf"]
            land = (rank > k) & ~unknown & ~kin[k]
            if sf["onto"]:
                land &= np.isin(M, [i + 1 for i, g in enumerate(grounds) if g["id"] in sf["onto"]])
            # Where a river comes out the waves die down: its water is pushing the other way.
            calm = np.ones((h, w), dtype=np.float32)
            if kin[k].any():
                # flat calm within a cell of the river, back to full strength some three cells off
                off = _reach(kin[k], ~kin[k], 48)
                calm = np.where(off > 0, np.clip((off - 16) / 30.0, 0, 1), 1.0).astype(np.float32)
            surfs.append({"look": sf, "out": _reach((rank > k) & ~kin[k], mine, 12), "up": _reach(mine, land, sf["reach"] + 1),
                          "calm": calm,
                          "when": _noise(X, Y, 70, s + 900).astype(np.float32),
                          "chance": _hash(X, Y, s + 901).astype(np.float32)})
        for wv in look.get("water", []):
            waters.append(_water_motion(wv, k, mine, G, X, Y, M, grounds, joined, s, fixed, VX, VY, look.get("flows")))
        if look["shore"]:
            higher = (rank > k) & ~kin[k]
            if look["shore"].get("lap"):
                laps.append((mine, look["shore"]["colors"], higher))
            else:
                reach = higher
                for band in look["shore"]["colors"]:
                    grown = _grow(reach)
                    idx[grown & ~reach & mine] = P[band]
                    reach = grown
                fixed |= reach & mine
        if look["rim"]:
            lower = np.zeros((h, w), dtype=bool)
            apart = np.where(kin[k], k, rank)  # kin are not "lower": no rim against them
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                below = _shift(apart, dx, dy, fill=k) < k
                if still_water is not None:
                    below &= _shift(still_water, dx, dy, fill=False)
                lower |= below
            idx[mine & lower] = P[look["rim"]]
            fixed |= mine & lower

    # What each ground spills onto the lower ones around it. After every surface is drawn, and
    # in rising order, so a higher ground's apron lies over a lower one's.
    apron = np.zeros((h, w), dtype=bool)
    for k in present:
        if not 0 < k <= len(grounds) or not grounds[k - 1]["look"].get("apron"):
            continue
        look, lower = grounds[k - 1]["look"], (rank < k) & ~void & ~kin[k]
        width = int(max(z["width"] + z["wobble"] for z in look["apron"])) + 1
        def only(z):
            if not z.get("over"):
                return True
            return np.isin(M, [i + 1 for i, t in enumerate(grounds) if t["id"] in z["over"]])

        apron |= _zone(idx, _reach(M == k, lower, width), look["apron"], X, Y, _seed(grounds[k - 1]["id"]) + 500, P,
                       lower, only)
    fixed |= apron

    cliff = np.zeros((h, w), dtype=bool)
    for k in present:  # ascending, so a higher material's face covers a lower one's
        if not 0 < k <= len(grounds) or not grounds[k - 1]["look"]["cliff"]:
            continue
        cl, mine = grounds[k - 1]["look"]["cliff"], M == k
        over = (rank < k) & ~kin[k]
        if still_water is not None:
            over &= still_water
        if cl["over"]:
            ids = {i + 1 for i, g in enumerate(grounds) if g["id"] in cl["over"]}
            over &= np.isin(M, list(ids))
        done = np.zeros((h, w), dtype=bool)
        cracked = _hash(X, Y * 0, _seed(grounds[k - 1]["id"]) + 5) < 0.2 if cl["cracks"] else None
        for d, name in enumerate(cl["colors"], start=1):
            face = _shift(mine, 0, d) & over & ~done
            idx[face] = P[name]
            if cracked is not None:
                idx[face & cracked] = P.dark[P[name]]
            done |= face
        cliff |= done
    # Steps: the land cut into tiers by height, a face below each step, a lit lip above it.
    if RF is not None:
        # Which step each cell is on. A lone cell on a step of its own is smoothed into its
        # neighbours: a single raised square reads as a mistake, not as land.
        tc = np.floor(RF / TIER).astype(np.int16)
        for _ in range(2):
            pad = np.pad(tc, 1, mode="edge")
            near4 = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
            alone = (near4 != tc).all(axis=0)
            tc = np.where(alone, np.median(near4, axis=0).astype(np.int16), tc)
        cells_big = np.repeat(np.repeat(tc, C, 0), C, 1)
        # Each step claims its pixels the way grounds do, but with little blur and less wander:
        # edges keep to the cell borders, corners come out round, the line shakes by a pixel.
        best = np.full((h, w), -9.0, dtype=np.float32)
        tier = np.zeros((h, w), dtype=np.int16)
        shake = (1 - SNAP_STEP) * (1.2 * (_noise(X, Y, 9, 8081) - 0.5)) + 0.12 * (_noise(X, Y, 5, 8083) - 0.5)
        for t in np.unique(tc).tolist():
            near = Image.fromarray(((cells_big == t) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.6))
            f = np.asarray(near, dtype=np.float32) / 255 + shake * (1 if t % 2 else -1) + t * 1e-4
            win = f > best
            best[win] = f[win]
            tier[win] = t
        level_ground = np.zeros(len(grounds) + 2, dtype=bool)   # still water lies level: no steps in it
        faces = np.zeros((len(grounds) + 2, 6), dtype=np.int16)
        lips = np.zeros(len(grounds) + 2, dtype=np.int16)
        for i, t in enumerate(grounds, start=1):
            look = t["look"]
            base = P[look["base"]]
            face = [P[c] for c in look.get("face") or []] or [P.dark[base], P.dark[P.dark[base]]]
            faces[i] = [face[min(j * len(face) // 6, len(face) - 1)] for j in range(6)]
            lips[i] = P[look["lip"]] if look.get("lip") else base
            level_ground[i] = t["pass"] == "swim" and not look.get("flows")
        tier = np.where(level_ground[np.clip(M, 0, len(grounds) + 1)], -999, tier)
        # Only cliffs are drawn. Masks in cells: a cliff along the top edge of a cell, a ramp
        # there, a cliff at its side; brought to pixels and widened a little for round corners.
        def px(mask, grow=0):
            big = np.repeat(np.repeat(mask, C, 0), C, 1)
            for _ in range(grow):
                big = big | _shift(big, 1, 0) | _shift(big, -1, 0)
            return big

        cliff_top, ramp_top, cliff_side = CL
        running = np.isin(M, [i for i, t in enumerate(grounds, start=1) if t["look"].get("flows")])
        zone_face = px(cliff_top | ramp_top, 2) | running        # a river falls white over every step it takes
        zone_ramp = px(ramp_top, 1)
        zone_lip = px(np.roll(cliff_top, -1, 0) & (np.arange(cliff_top.shape[0]) < cliff_top.shape[0] - 1)[:, None], 2)
        zone_side = px(cliff_side, 1)
        done = np.zeros((h, w), dtype=bool)
        # Where running water goes over a step it is a fall, and a fall moves: remember, for every
        # pixel of the face it falls down, how far down it is, how tall the face is and whose it is.
        flowing = np.array([False] + [bool(t["look"].get("flows")) for t in grounds] + [False])
        rocky = next((i for i, t in enumerate(grounds, start=1) if t["id"] == "mountain"), None)
        rock_face = faces[rocky] if rocky else np.array([P["s2"], P["s1"], P["s1"], P["s0"], P["s0"], P["n0"]], dtype=np.int16)
        fall_d, fall_h, fall_t = (np.zeros((h, w), dtype=np.int16) for _ in range(3))
        for d in range(1, FACE * 4 + 1):
            above = _shift(tier, 0, d, fill=-999)
            drop = above - tier
            face = (drop > 0) & (tier > -999) & (d <= FACE * drop) & ~done & zone_face
            if not face.any():
                continue
            top = _shift(M, 0, d)
            # A fall needs water at the top AND the water going over: water below it too, the
            # same stream running south across the step (faces are only on the south side).
            # Water on top with no water coming down is a wall of rock; a step that a river
            # merely runs along has no face at all - its surface is level across.
            wet_top = face & flowing[np.clip(top, 0, len(grounds) + 1)]
            wet_here = flowing[np.clip(M, 0, len(grounds) + 1)]
            south = (VY > 0.3) if VY is not None else np.zeros((h, w), dtype=bool)
            fall = wet_top & wet_here & south
            along = wet_top & wet_here & ~south
            face &= ~along
            dry = wet_top & ~wet_here
            fall_d[fall], fall_h[fall], fall_t[fall] = d, (FACE * drop)[fall], top[fall]
            row = np.clip((d - 1) * 6 // np.maximum(FACE * drop, 1), 0, 5)
            idx[face] = faces[np.clip(top[face], 0, len(grounds) + 1), row[face]]
            idx[dry] = rock_face[row[dry]]
            stair = face & zone_ramp & ((d // 2) % 2 == 0)                 # a ramp: the face cut into treads
            idx[stair] = lips[np.clip(top[stair], 0, len(grounds) + 1)]
            done |= face
        first = (_shift(tier, 0, 1, fill=-999) > tier) & (tier > -999)
        # the lit lip: along cliffs, and in a river only where it goes over a fall
        lip = _shift(first, 0, -1) & ~done & (zone_lip | (running & _shift(fall_d == 1, 0, -1)))
        idx[lip] = lips[np.clip(M[lip], 0, len(grounds) + 1)]
        side = np.zeros((h, w), dtype=bool)
        for dx in (1, -1):
            side |= (_shift(tier, dx, 0, fill=-999) > tier) & (tier > -999)
        side &= ~done & ~lip & zone_side
        idx[side] = P.dark[idx[side]]
        cliff |= done
        fixed |= done | lip
        is_fall = fall_d > 0
        if is_fall.any():
            # the plunge pool: a few pixels of the water at the foot of the fall
            foot = np.zeros((h, w), dtype=bool)
            for k in range(1, 5):
                foot |= _shift(is_fall, 0, k) & (k <= 2 + (_hash(X, Y, 919) < 0.5))
            foot &= flowing[np.clip(M, 0, len(grounds) + 1)] & ~is_fall & ~done
            light = np.zeros(len(grounds) + 2, dtype=np.int16)
            mid = np.zeros(len(grounds) + 2, dtype=np.int16)
            for i, t in enumerate(grounds, start=1):
                cols = [P[c] for c in t["look"].get("face") or []] or [P[t["look"]["base"]]]
                by_light = sorted(set(cols), key=lambda c: -int(P.rgba[c][:3].astype(int).sum()))
                light[i] = by_light[0]                          # the white water
                mid[i] = by_light[1] if len(by_light) > 1 else by_light[0]
            fy, fx = np.nonzero(is_fall)
            py, px_ = np.nonzero(foot)
            under = _shift(M, 0, -1)   # the water a foot pixel belongs to is its own ground
            falls = {"y": fy, "x": fx, "wx": X[fy, fx], "wy": Y[fy, fx], "d": fall_d[fy, fx], "h": fall_h[fy, fx],
                     "light": light[np.clip(fall_t[fy, fx], 0, len(grounds) + 1)],
                     "mid": mid[np.clip(fall_t[fy, fx], 0, len(grounds) + 1)],
                     "still": idx[fy, fx].copy(),
                     "py": py, "px": px_, "pwx": X[py, px_], "pwy": Y[py, px_],
                     "plight": light[np.clip(M[py, px_], 0, len(grounds) + 1)],
                     "pmid": mid[np.clip(M[py, px_], 0, len(grounds) + 1)]}
            fixed |= foot
        else:
            falls = None
    else:
        falls = None

    # What grows on the ground: a cover over its cells, its edge wandering past them the way a
    # ground's does (further for long grass), packed with blades that stand up from their roots
    # - so long grass stands over the cell above it. Not on water, not on a cliff face.
    blades = None
    if PL is not None and PL[0].any():
        codes, plants = PL
        bigp = np.repeat(np.repeat(codes, C, 0), C, 1)
        land_px = ~np.isin(M, [i for i, t in enumerate(grounds, start=1) if t["pass"] == "swim"]) & ~cliff & ~void
        roots = []
        for kp in [int(v) for v in np.unique(codes) if v]:
            gr = plants[kp - 1]["grow"]
            near = Image.fromarray(((bigp == kp) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.0 + 0.5 * gr["spill"]))
            s = _seed(plants[kp - 1]["id"])
            claim = (np.asarray(near, dtype=np.float32) / 255 + (0.9 * (_noise(X, Y, 9, s) - 0.5) + 0.5 * (_noise(X, Y, 4, s + 3) - 0.5)) * 0.6
                     + 0.03 * gr["spill"])
            cover = (claim > 0.5) & land_px
            tall = gr["height"][1] >= 6           # long grass hides the ground; short grass lets it (and its flowers) show
            idx[cover & (_hash(X, Y, s + 11) < (0.7 if tall else 0.2))] = P[gr["under"]]
            if tall:
                fixed |= cover
            pick = cover & (_hash(X, Y, s + 12) < gr["density"])
            ys, xs = np.nonzero(pick)
            lo, hi = gr["height"]
            # a blade's height: shorter toward the edge of the cover, so a patch thins out
            edge = np.clip((np.asarray(near, dtype=np.float32)[ys, xs] / 255 - 0.3) / 0.5, 0.35, 1)
            ln = np.maximum(1, np.rint((lo + _hash(X[ys, xs], Y[ys, xs], s + 13) * (hi - lo)) * edge)).astype(np.int16)
            # some blades of a flowering grass end in a flower, its colour picked per blade
            bl = gr.get("bloom") or []
            flower = np.full(len(ys), -1, dtype=np.int16)
            if bl:
                # fewer toward the edge of the patch, so flowers thin out instead of stopping at a line
                wide_ = np.asarray(Image.fromarray(((bigp == kp) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(7)),
                                   dtype=np.float32)[ys, xs] / 255
                fade = np.clip((wide_ - 0.15 + 0.35 * (_noise(X, Y, 11, s + 18)[ys, xs] - 0.5)) / 0.6, 0, 1)
                opens = _hash(X[ys, xs], Y[ys, xs], s + 15) < gr.get("blooming", 0.15) * fade * (
                    0.4 + 0.9 * _noise(X, Y, 20, s + 17)[ys, xs])
                which = (_hash(X[ys, xs], Y[ys, xs], s + 16) * len(bl)).astype(np.int64)
                flower = np.where(opens, np.array([P[c] for c in bl], dtype=np.int16)[which], -1).astype(np.int16)
            n_ = len(ys)
            form = gr.get("form", "blades")
            side = np.zeros(n_, dtype=np.int8)
            if form == "fronds":             # each root: a leaf arching out to either side
                ys, xs, ln, flower = (np.concatenate([a, a]) for a in (ys, xs, ln, flower))
                side = np.concatenate([np.full(n_, -1, dtype=np.int8), np.ones(n_, dtype=np.int8)])
                n_ *= 2
            motion = 0 if gr.get("still") else 1 if gr.get("stiff") else 2     # how far the wind may push it
            roots.append((ys, xs, ln, np.array([P[c] for c in gr["colors"]], dtype=np.int16),
                          _hash(X[ys, xs], Y[ys, xs], s + 14) < 0.5, flower,
                          np.full(n_, gr.get("width", 1), dtype=np.int8), side, np.full(n_, motion, dtype=np.int8),
                          # a crust is one pixel high: each patch takes its own colour from the list instead
                          (_hash(X[ys, xs], Y[ys, xs], s + 19) * len(gr["colors"])).astype(np.int8) if form == "crust"
                          else np.zeros(n_, dtype=np.int8)))
        if roots:
            ys = np.concatenate([r[0] for r in roots])
            order = np.argsort(ys, kind="stable")       # back to front: a nearer blade covers a further one
            cols = np.zeros((len(ys), 6), dtype=np.int16)
            nc = np.zeros(len(ys), dtype=np.int16)
            at_ = 0
            for r in roots:
                n_ = len(r[0])
                cols[at_:at_ + n_, :len(r[3])] = r[3]
                nc[at_:at_ + n_] = len(r[3])
                at_ += n_
            blades = {"y": ys[order], "x": np.concatenate([r[1] for r in roots])[order],
                      "len": np.concatenate([r[2] for r in roots])[order], "cols": cols[order], "nc": nc[order],
                      "left": np.concatenate([r[4] for r in roots])[order],
                      "flower": np.concatenate([r[5] for r in roots])[order],
                      "width": np.concatenate([r[6] for r in roots])[order],
                      "side": np.concatenate([r[7] for r in roots])[order],
                      "motion": np.concatenate([r[8] for r in roots])[order],
                      "pick": np.concatenate([r[9] for r in roots])[order]}
            blades["wind"] = (((blades["x"] + X[0, 0]) * WIND[0] + (blades["y"] + Y[0, 0]) * WIND[1]) / GUST_SPAN
                              + 0.5 * _noise(X, Y, 90, 4711)[blades["y"], blades["x"]]).astype(np.float32)

    wind = None
    if gusts or any(wm["move"] == "ruffle" for wm in waters):  # where each pixel is along the train of gusts; the noise makes their fronts ragged
        wind = ((X * WIND[0] + Y * WIND[1]) / GUST_SPAN + 0.5 * _noise(X, Y, 90, 4711)).astype(np.float32)
    return {"M": M, "cliff": cliff, "idx": idx, "fixed": fixed | cliff, "specks": specks, "laps": laps, "apron": apron,
            "gusts": gusts, "wind": wind, "surfs": surfs, "streams": streams, "falls": falls, "blades": blades,
            "waters": waters}


def _water_motion(wv: dict, k: int, mine, G, X, Y, M, grounds, joined, s: int, fixed, VX=None, VY=None,
                  flows=False) -> dict:
    """What one ground's own water motion needs that does not change from frame to frame."""
    h, w = mine.shape
    out = {"move": wv["move"], "wv": wv}
    if wv["move"] == "surface":
        # The whole surface is covered in wavelets, the way grass is covered in blades: little
        # flat marks, each catching the light and losing it again. Their moments are staggered
        # along the way the water goes - downwind, or down a river's current - so bands of
        # them travel across it.
        free = mine & ~fixed
        ys, xs = np.nonzero(free & (_hash(X, Y, s + 1501) < wv["density"]))
        ux, uy = np.full(len(ys), WIND[0], dtype=np.float32), np.full(len(ys), WIND[1], dtype=np.float32)
        if flows and VX is not None:
            vx, vy = VX[ys, xs], VY[ys, xs]
            sp_ = np.hypot(vx, vy)
            run = sp_ > 0.2
            ux, uy = np.where(run, vx / np.maximum(sp_, 1e-6), ux), np.where(run, vy / np.maximum(sp_, 1e-6), uy)
        n = np.hypot(ux, uy)
        ux, uy = ux / n, uy / n
        wx, wy = X[ys, xs], Y[ys, xs]
        phase = ((wx * ux + wy * uy) / wv["spacing"] + 0.6 * _noise(X, Y, 23, s + 1502)[ys, xs]
                 + 0.25 * _hash(wx, wy, s + 1503)).astype(np.float32)
        out.update({"ys": ys, "xs": xs, "ux": ux, "uy": uy, "phase": phase, "free": free,
                    "lucky": _hash(wx, wy, s + 1504) < 0.25})
    elif wv["move"] == "swell":
        # How far each pixel is from the land, measured between cell centres and smoothed back
        # to pixels: the crests follow the run of the coast, not every nook of it.
        land = (G > k) & (G <= len(grounds)) & ~np.isin(G, list(joined[k]))
        far = float(SWELL_REACH + 1)
        dc = np.full(G.shape, far, dtype=np.float32)
        r = SWELL_REACH
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                dd = math.hypot(dx, dy)
                if dd <= r:
                    dc = np.where(_shift(land, dx, dy, fill=False), np.minimum(dc, dd), dc)
        fx, fy = (X - X[0, 0] + 0.5) / C - 0.5, (Y - Y[0, 0] + 0.5) / C - 0.5
        x0, y0 = np.clip(np.floor(fx).astype(np.int64), 0, G.shape[1] - 1), np.clip(np.floor(fy).astype(np.int64), 0, G.shape[0] - 1)
        x1, y1 = np.clip(x0 + 1, 0, G.shape[1] - 1), np.clip(y0 + 1, 0, G.shape[0] - 1)
        ax, ay = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
        d = ((dc[y0, x0] * (1 - ax) + dc[y0, x1] * ax) * (1 - ay) + (dc[y1, x0] * (1 - ax) + dc[y1, x1] * ax) * ay) * C
        # crests come in, but where they break into dashes does not move: only those pixels
        # ever show one, so only they are looked at frame by frame
        fade = np.clip((far * C - d) / (far * C * 0.5), 0, 1) * np.clip((d - 12) / 8, 0, 1)
        brk = _noise(X, Y, 6, s + 1201)
        ys, xs = np.nonzero(mine & ~fixed & (brk > 1 - 0.42 * fade))
        out.update({"ys": ys, "xs": xs, "d": d[ys, xs].astype(np.float32),
                    "glints": (brk[ys, xs] > 1 - 0.1 * fade[ys, xs])})
    elif wv["move"] == "ruffle":
        chance = _hash(X, Y, s + 1301)
        ys, xs = np.nonzero(mine & ~fixed & ((chance < 0.06) | (chance > 1 - 0.012)))
        out.update({"ys": ys, "xs": xs, "chance": chance[ys, xs].astype(np.float32), "free": mine & ~fixed})
    else:  # rings: a few points, away from the edge, each at its own moment in the loop
        inner = mine & ~fixed
        for _ in range(3):
            inner &= ~_grow(~inner)
        ys, xs = np.nonzero(inner & (_hash(X, Y, s + 1401) < wv["density"]))
        out.update({"ys": ys, "xs": xs, "free": mine & ~fixed,
                    "t0": np.floor(_hash(X[ys, xs], Y[ys, xs], s + 1402) * FRAMES).astype(np.int64)})
    return out


# a ring a few px across, wider than tall (the ground is seen from above and a little south)
_RINGS = [[(0, 0)]] + [[(dx, dy) for dy in range(-r, r + 1) for dx in range(-r - 1, r + 2)
                        if round(math.hypot(dx / 1.25, dy / 0.8)) == r] for r in (1, 2, 3)]


def _water(idx, wm: dict, g: dict, frame: int, P: Palette) -> None:
    """One moment of a ground's own water motion."""
    wv = wm["wv"]
    h, w = idx.shape
    if wm["move"] == "surface":
        # each wavelet: a dot, a short dash, longer and drifting on a pixel, gone for a while;
        # two rounds a loop. A gust lengthens it and puts a glint in it.
        t = (wm["phase"] - frame * 2 / FRAMES) % 1.0
        blow = None
        if g.get("wind") is not None:
            if g.get("_blow", (None,))[0] != frame:
                g["_blow"] = (frame, gust_at(g["wind"], frame).astype(np.float32))
            blow = g["_blow"][1][wm["ys"], wm["xs"]]
        size = np.where(t < 0.12, 1, np.where(t < 0.3, 2, np.where(t < 0.48, 3, np.where(t < 0.6, 2, 0))))
        if blow is not None:
            size = np.where(size > 0, size + (blow > 0.35), 0)
        drift = np.where(t < 0.3, 0, 1)
        x0 = wm["xs"] + np.rint(wm["ux"] * drift).astype(np.int64) - size // 2
        y0 = wm["ys"] + np.rint(wm["uy"] * drift).astype(np.int64)
        col = P[wv["color"]]
        for i in range(4):
            on = size > i
            x, y = x0[on] + i, y0[on]
            ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            x, y = x[ok], y[ok]
            hit = wm["free"][y, x]
            idx[y[hit], x[hit]] = col
        if wv["glint"]:
            gl = (size >= 3) & ((t > 0.3) & (t < 0.48))
            if blow is not None:
                gl &= (blow > 0.2) | wm["lucky"]          # between gusts only a few catch the sun
            x, y = wm["xs"][gl], wm["ys"][gl]
            hit = wm["free"][y, x]
            idx[y[hit], x[hit]] = P[wv["glint"]]
    elif wm["move"] == "swell":
        # crests every `spacing` px, moving in by one spacing a loop; broken into dashes, fading
        # out to sea, and left to the surf near the land
        sp = wv["spacing"]
        on = np.floor(wm["d"] + frame * sp / FRAMES) % sp == 0
        idx[wm["ys"][on], wm["xs"][on]] = P[wv["color"]]
        if wv["glint"]:
            g_ = on & wm["glints"]
            idx[wm["ys"][g_], wm["xs"][g_]] = P[wv["glint"]]
    elif wm["move"] == "ruffle":
        if g["wind"] is None:
            return
        if g.get("_blow", (None,))[0] != frame:            # one wind for every water in the window
            g["_blow"] = (frame, gust_at(g["wind"], frame).astype(np.float32))
        ys, xs = wm["ys"], wm["xs"]
        blow = g["_blow"][1][ys, xs]
        sel = wm["chance"] < 0.06 * blow
        for ox, oy in SHAPES["h2"]:
            y, x = ys[sel] + oy, xs[sel] + ox
            ok = (x < w) & (y < h)
            y, x = y[ok], x[ok]
            hit = wm["free"][y, x]
            idx[y[hit], x[hit]] = P[wv["color"]]
        if wv["glint"]:
            gl = wm["chance"] > 1 - 0.012 * blow
            idx[ys[gl], xs[gl]] = P[wv["glint"]]
    else:
        free = wm["free"]
        age = (frame - wm["t0"]) % FRAMES
        for r, ring in enumerate(_RINGS):
            now = age // 2 == r                            # a dot, then each ring, two frames each
            ys, xs = wm["ys"][now], wm["xs"][now]
            colour = P[wv["glint"] or wv["color"]] if r == 0 else P[wv["color"]]
            for dx, dy in ring:
                y, x = ys + dy, xs + dx
                ok = (y >= 0) & (y < h) & (x >= 0) & (x < w)
                y, x = y[ok], x[ok]
                hit = free[y, x]
                idx[y[hit], x[hit]] = colour


def gust_at(u, frame: int):
    """How hard the wind is blowing (0..1) at positions `u` along the train of gusts: nothing
    between gusts, strongest at a gust's front, dying away behind it."""
    band = (u - frame / FRAMES) % 1.0
    return np.where(band < GUST_WIDTH, band / GUST_WIDTH, 0.0)


def _surf(idx, sf: dict, g: dict, frame: int, P: Palette) -> None:
    """One moment of the waves on one sea. `when` shifts the timing along the shore."""
    look, out, up, chance = sf["look"], sf["out"], sf["up"], sf["chance"]
    R = look["reach"] * sf["calm"]
    phi = (frame / FRAMES + sf["when"]) % 1.0
    foam, water = P[look["foam"]], P[look["water"]]
    # coming in: a broken line of foam, nearer the land each frame, a paler swell behind it
    line = np.rint(11 * (1 - phi / 0.45)).astype(np.int8) + 1
    coming = (phi < 0.45) & (out > 0) & (sf["calm"] > 0.6)
    idx[coming & (out == line + 2) & (chance > 0.35)] = water
    idx[coming & (out == line) & (chance > 0.2)] = foam
    # on the land: up fast, back slow, and the sand stays dark a while after
    run = np.where(phi < 0.45, 0.0, np.where(phi < 0.6, (phi - 0.45) / 0.15, np.where(phi < 0.9, 1 - (phi - 0.6) / 0.3, 0.0)))
    wet = np.where(phi < 0.45, 0.0, np.where(phi < 0.6, run, np.where(phi < 0.82, 1.0, 1 - (phi - 0.82) / 0.18)))
    land = (up > 0) & ~g["cliff"]
    damp = land & (up <= np.ceil(wet * R))
    idx[damp] = P.dark[idx[damp]]
    cover = np.ceil(run * R).astype(np.int8)
    under = land & (up <= cover)
    idx[under] = water
    idx[under & (chance < 0.12)] = foam
    idx[land & (cover > 0) & (up == cover)] = foam
    # just behind the front on the way back, a thin trace of foam left on the wet sand
    idx[land & (phi >= 0.6) & (cover > 0) & (up == cover + 1) & (up <= R) & (chance < 0.3)] = foam


def _blades(idx, b: dict, frame: int) -> None:
    """Everything that grows, stem by stem, root to tip, upward. A frond arches out to its side;
    a tuft or crust is wider than a stem. Where a gust is on it the upper half is laid over with
    the wind - less for a stiff stem, not at all for a crust - and now and then it stirs."""
    h, w = idx.shape
    blow = gust_at(b["wind"], frame)
    push = np.where(blow > 0.25, np.where(blow > 0.6, 2, 1), 0) * (1 if WIND[0] >= 0 else -1)
    lean = np.clip(push, -b["motion"], b["motion"])
    sway = np.where(b["left"], -1, 1) * (((frame + (b["x"] * 7 + b["y"] * 3)) // 3) % 4 == 0) * (b["motion"] == 2)
    top = int(b["len"].max()) if len(b["len"]) else 0
    rows = np.arange(len(b["len"]))
    for k in range(top):
        on = b["len"] > k
        if not on.any():
            continue
        frac = k / np.maximum(b["len"], 1)
        bend = np.where(frac >= 0.5, np.where(lean != 0, lean, sway), 0) * (frac >= 0.5)
        bend = np.where(frac >= 0.8, bend * (1 + (lean != 0)), bend).astype(np.int64)
        arch = (b["side"] * ((k * k * 1.6) // np.maximum(b["len"], 1))).astype(np.int64)   # a frond curves out
        y = b["y"] - k
        x = b["x"] + bend + arch
        band = np.minimum((frac * b["nc"]).astype(np.int64), b["nc"] - 1)
        c = b["cols"][rows, band]
        across = np.where(frac < 0.6, b["width"], np.maximum(b["width"] - 1, 1))   # a cushion rounds off at the top
        for j in range(int(b["width"].max())):
            ok = on & (j < across) & (y >= 0) & (x + j >= 0) & (x + j < w)
            idx[y[ok], x[ok] + j] = c[ok]
        # a flower or seed head at the tip: the last pixel, and a petal beside it on taller stems
        tip = (b["len"] == k + 1) & (b["flower"] >= 0) & on & (y >= 0) & (x >= 0) & (x < w)
        if tip.any():
            idx[y[tip], x[tip]] = b["flower"][tip]
            wide = tip & (b["len"] >= 3) & (x + 1 < w)
            idx[y[wide], x[wide] + 1] = b["flower"][wide]


def _falls(idx, f: dict, frame: int) -> None:
    """Water going over a step: white streaks running down the curtain, each column at its own
    pace (two speeds, so the loop still closes), a white lip at the top and white water at the
    bottom; below it the pool churns."""
    col = _hash(f["wx"], 0, 3301)
    speed = np.where(col < 0.5, 2, 4)                       # px a frame; 24 frames x 2 or 4 is a whole number of 8s
    m = (f["wy"] + (col * 8).astype(np.int64) - frame * speed) % 8
    colour = np.where(m < 2, f["light"], np.where(m < 4, f["mid"], f["still"]))
    colour = np.where((f["d"] <= 1) | (f["d"] >= f["h"]), f["light"], colour)   # the lip, and where it hits
    idx[f["y"], f["x"]] = colour
    churn = _hash(f["pwx"], f["pwy"] * 7 + frame % CYCLE, 4421)
    idx[f["py"], f["px"]] = np.where(churn < 0.45, f["plight"], f["pmid"])


def _ground_frame(g: dict, frame: int, P: Palette):
    """The ground at one moment: the still picture plus whatever is moving on it."""
    idx = g["idx"].copy()
    for mine, sp, seeds, phase in g["specks"]:
        free = mine & ~g["fixed"]
        ph = (phase + frame) % CYCLE
        if sp["anim"] == "twinkle":
            parts = [(seeds & (ph < CYCLE // 2), 0)]
        else:  # drift: appears, slides two pixels, is gone for a beat
            parts = [(seeds & (ph // 2 == d), d) for d in range(3)]
        for sel, dx in parts:
            for ox, oy in SHAPES[sp["shape"]]:
                idx[_shift(sel, ox + dx, oy) & free] = P[sp["color"]]
    h, w = idx.shape
    for st in g["streams"]:
        free = st["mine"] & ~g["fixed"]
        step = (st["ph"] + frame) % CYCLE
        speed = np.hypot(st["vx"], st["vy"])
        still = speed < 0.2
        ux = np.where(still, 1.0, st["vx"] / np.maximum(speed, 1e-6))
        uy = np.where(still, 0.0, st["vy"] / np.maximum(speed, 1e-6))
        far = step * np.where(still, 0.5, 1.6 * st["fast"])
        x0, y0 = st["xs"] + np.rint(ux * far).astype(np.int64), st["ys"] + np.rint(uy * far).astype(np.int64)
        along_x = np.abs(ux) >= np.abs(uy)  # a streak lies along the way it travels
        for i in range(len(SHAPES[st["sp"]["shape"]])):
            x, y = x0 + np.where(along_x, i, 0), y0 + np.where(along_x, 0, i)
            ok = (step < CYCLE - 2) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
            x, y = x[ok], y[ok]
            hit = free[y, x]
            idx[y[hit], x[hit]] = P[st["sp"]["color"]]
    if g["gusts"]:
        if g.get("_blow", (None,))[0] != frame:
            g["_blow"] = (frame, gust_at(g["wind"], frame).astype(np.float32))
        blow = g["_blow"][1]
        for mine, sp, chance in g["gusts"]:
            sel = (chance < sp["density"] * blow) & mine & ~g["fixed"]
            for ox, oy in SHAPES[sp["shape"]]:
                idx[_shift(sel, ox, oy) & mine & ~g["fixed"]] = P[sp["color"]]
    for wm in g.get("waters", []):
        _water(idx, wm, g, frame, P)
    if g.get("falls"):
        _falls(idx, g["falls"], frame)
    if g.get("blades") is not None:
        _blades(idx, g["blades"], frame)
    for sf in g["surfs"]:
        _surf(idx, sf, g, frame, P)
    for mine, colors, higher in g["laps"]:
        order = list(colors)
        if _LAP[frame % CYCLE] and len(order) > 1:
            order[0], order[1] = order[1], order[0]
        reach = higher
        for band in order:
            grown = _grow(reach)
            idx[grown & ~reach & mine & ~g["cliff"] & ~g["apron"]] = P[band]
            reach = grown
    return idx


# ---------------------------------------------------------------- the things standing on it

def _blit(idx, sprite, left: int, top: int) -> None:
    h, w = idx.shape
    sh, sw = sprite.shape
    x0, y0, x1, y1 = max(left, 0), max(top, 0), min(left + sw, w), min(top + sh, h)
    if x0 >= x1 or y0 >= y1:
        return
    part = sprite[y0 - top:y1 - top, x0 - left:x1 - left]
    view = idx[y0:y1, x0:x1]
    view[part >= 0] = part[part >= 0]


def _place(m, tiles, grounds, g: dict, gx0: int, gy0: int, with_objects: bool) -> list:
    """Where every object and piece of decor in the window stands, in window pixels: which
    variant, mirrored or not, where its feet are, where it is in the loop. Back to front."""
    h, w = g["M"].shape
    code = {t["id"]: i + 1 for i, t in enumerate(grounds)}
    out = []
    for cy in range(max(gy0, 0), min(gy0 + h // C, m.height)):
        for cx in range(max(gx0, 0), min(gx0 + w // C, m.width)):
            obj = m.layers["object"][cy][cx]
            px, py = (cx - gx0) * C, (cy - gy0) * C
            plant = m.layers["plant"][cy][cx] if "plant" in m.layers else None
            pt = tiles.get(plant) if plant else None
            if obj:
                if not with_objects:
                    continue
                t = tiles.get(obj)
                if t is None or t["layer"] != "object":
                    out.append({"tile": None, "fx": px + C // 2, "fy": py + C - 1, "cell": (cx, cy)})
                    continue
                s, j = _seed(obj), int(t.get("jitter", 0))
                out.append({
                    "tile": obj, "cell": (cx, cy), "decor": False,
                    "variant": int(_hash(cx, cy, s) * len(t["variants"])),
                    "flip": bool(t.get("flip")) and _hash(cx, cy, s + 1) < 0.5,
                    "fx": px + C // 2 + round((_hash(cx, cy, s + 2) * 2 - 1) * j),
                    "fy": py + C - 2 + round((_hash(cx, cy, s + 3) * 2 - 1) * (j // 2)),
                    "phase": int(_hash(cx, cy, s + 9) * CYCLE), "shadow": bool(t.get("shadow"))})
                continue
            gid = m.layers["ground"][cy][cx]
            t = tiles.get(gid)
            if t is None or t["layer"] != "ground" or (pt is not None and pt["grow"]["height"][1] >= 6):   # long grass hides its decor
                continue
            for i, rule in enumerate(t["look"]["decor"]):
                d = tiles.get(rule["tile"])
                if d is None or d["layer"] != "object" or _hash(cx, cy, _seed(gid) + 977 * (i + 1)) >= rule["density"]:
                    continue
                s = _seed(rule["tile"]) + i
                fx = px + 3 + int(_hash(cx, cy, s + 5) * (C - 6))
                fy = py + 4 + int(_hash(cx, cy, s + 6) * (C - 5))
                # Only where the foot really is on this material: borders wander.
                if not (0 <= fx < w and 0 <= fy < h) or g["M"][fy, fx] != code[gid] or g["cliff"][fy, fx]:
                    continue
                out.append({"tile": rule["tile"], "cell": (cx, cy), "decor": True,
                            "variant": int(_hash(cx, cy, s + 4) * len(d["variants"])),
                            "flip": bool(d.get("flip")) and _hash(cx, cy, s + 7) < 0.5,
                            "fx": fx, "fy": fy, "phase": int(_hash(cx, cy, s + 9) * CYCLE), "shadow": False})
                break
    for d in out:  # where it stands along the train of gusts, for things the wind pushes over
        wx, wy = np.array([[d["fx"] + gx0 * C]], dtype=np.int64), np.array([[d["fy"] + gy0 * C]], dtype=np.int64)
        d["wind"] = float((wx[0, 0] * WIND[0] + wy[0, 0] * WIND[1]) / GUST_SPAN + 0.5 * _noise(wx, wy, 90, 4711)[0, 0])
    out.sort(key=lambda d: (d["fy"], d["fx"]))
    return out


class _Sprites:
    """Sprites as index arrays, per tile / variant / mirroring / frame, made on demand."""

    def __init__(self, tiles, P):
        self.tiles, self.P, self.cache = tiles, P, {}

    def get(self, tile: str, variant: int, flip: bool, frame: int = 0):
        t = self.tiles[tile]
        motion = t.get("motion", "none")
        key = (tile, variant, flip, 0 if motion == "none" else frame if motion == "gust" else frame % CYCLE)
        if key not in self.cache:
            base = (tile, variant)
            if base not in self.cache:
                self.cache[base] = self.P.sprite(t["variants"][variant])
            spr = moved(self.cache[base], motion, key[3])
            self.cache[key] = spr[:, ::-1] if flip else spr
        return self.cache[key]


def _shadows(places, sprites: _Sprites, shape) -> np.ndarray:
    h, w = shape
    shadow = np.zeros((h, w), dtype=bool)
    for d in places:
        if not d.get("shadow"):
            continue
        spr = sprites.get(d["tile"], d["variant"], False)
        tall, wide = spr.shape
        # a pool of shade at the foot, drawn out away from the sun by the thing's height
        ox, oy = SUN["x"] * tall, SUN["y"] * tall
        rx = max(3.0, wide * 0.34 + abs(ox) * 0.5)
        ry = max(1.5, wide * 0.15 + abs(oy) * 0.5)
        cx, cy = d["fx"] + ox * 0.5, d["fy"] + oy * 0.5
        bx0, bx1 = max(0, int(cx - rx) - 1), min(w, int(cx + rx) + 3)
        by0, by1 = max(0, int(cy - ry) - 1), min(h, int(cy + ry) + 2)
        if bx0 < bx1 and by0 < by1:
            ys, xs = np.ogrid[by0:by1, bx0:bx1]
            shadow[by0:by1, bx0:bx1] |= ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0
    return shadow


def _draw_things(idx, places, sprites: _Sprites, shadow, frame: int, P: Palette) -> None:
    idx[shadow] = P.dark[idx[shadow]]
    for d in places:
        if d["tile"] is None:
            _blit(idx, np.full((C, C), P.missing, dtype=np.int16), d["fx"] - C // 2, d["fy"] - C + 1)
            continue
        if sprites.tiles[d["tile"]].get("motion") == "gust":
            spr = sprites.get(d["tile"], d["variant"], d["flip"], int(gust_at(d["wind"], frame) > 0.25))
        else:
            spr = sprites.get(d["tile"], d["variant"], d["flip"], frame + d["phase"])
        _blit(idx, spr, d["fx"] - spr.shape[1] // 2, d["fy"] - spr.shape[0] + 1)


# ---------------------------------------------------------------- pictures

_WHOLE = {}   # (what, the map's content) -> worked out over the whole map; a map is drawn in many pieces


def _whole(what, m, tiles, fn):
    """`fn(m, tiles)`, remembered for as long as the map and the ground tiles stay as they are:
    cliffs and currents are worked out over the whole map, and every piece of it needs them."""
    key = (what, m.width, m.height, tuple(map(tuple, m.layers["ground"])),
           tuple(map(tuple, m.relief)) if getattr(m, "relief", None) else None,
           tuple(map(tuple, getattr(m, "ramps", []))), tuple(getattr(m, "scale", ())),
           tuple(sorted((t["id"], t.get("pass"), bool(t.get("look", {}).get("flows")), t.get("rank", 0))
                        for t in tiles.values() if t["layer"] == "ground")))
    if key not in _WHOLE:
        if len(_WHOLE) > 16:
            _WHOLE.clear()
        _WHOLE[key] = fn(m, tiles)
    return _WHOLE[key]


def _window(m, tiles, pal, region):
    x, y, w, h = region or (0, 0, m.width, m.height)
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(m.width, int(x) + int(w)), min(m.height, int(y) + int(h))
    if x0 >= x1 or y0 >= y1:
        raise MapError("区域在地图之外")
    cw, ch = x1 - x0, y1 - y0
    P = Palette(pal)
    grounds = sorted((t for t in tiles.values() if t["layer"] == "ground"), key=lambda t: t.get("rank", 0))
    code = {t["id"]: i + 1 for i, t in enumerate(grounds)}  # 0 is bare; len+1 is a missing tile
    gx0, gy0, gw, gh = x0 - PAD, y0 - PAD, cw + 2 * PAD, ch + 2 * PAD
    G = np.zeros((gh, gw), dtype=np.int16)
    rows = m.layers["ground"]
    for j in range(gh):
        row = rows[min(max(gy0 + j, 0), m.height - 1)]  # past the edge the map just continues
        for i in range(gw):
            t = row[min(max(gx0 + i, 0), m.width - 1)]
            G[j, i] = code.get(t, len(grounds) + 1) if t else 0
    X, Y = np.meshgrid(np.arange(gx0 * C, (gx0 + gw) * C, dtype=np.int32),
                       np.arange(gy0 * C, (gy0 + gh) * C, dtype=np.int32))
    _XY[:] = [X, Y, X.astype(np.uint32), Y.astype(np.uint32)]
    _HASHED.clear()
    VX = VY = None
    flow = _whole("currents", m, tiles, currents)
    if flow is not None:
        ys = np.clip(np.arange(gy0, gy0 + gh), 0, m.height - 1)
        xs = np.clip(np.arange(gx0, gx0 + gw), 0, m.width - 1)
        def spread(part):  # cells to pixels, softened; through 8 bits, which is plenty for a direction
            big = np.repeat(np.repeat(part[np.ix_(ys, xs)], C, 0), C, 1)
            soft = Image.fromarray(np.rint((big + 1) * 127.5).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5))
            out = np.asarray(soft, dtype=np.float32) / 127.5 - 1
            return np.where(np.abs(out) < 0.02, 0, out).astype(np.float32)

        VX, VY = spread(flow[0]), spread(flow[1])
    RF = CL = None
    if getattr(m, "relief", None):
        rows_ = np.clip(np.arange(gy0, gy0 + gh), 0, m.height - 1)
        cols_ = np.clip(np.arange(gx0, gx0 + gw), 0, m.width - 1)
        RF = np.asarray(m.relief, dtype=np.float32)[np.ix_(rows_, cols_)]
        walls, ramps = _whole("cliffs", m, tiles, mapkit.cliffs), {tuple(r) for r in m.ramps}
        cliff_top, ramp_top, cliff_side = (np.zeros((gh, gw), dtype=bool) for _ in range(3))
        for j in range(gh):
            y = gy0 + j
            for i in range(gw):
                x = gx0 + i
                if not (0 <= x < m.width and 0 <= y < m.height):
                    continue
                cliff_top[j, i] = (x, y - 1, 0, 1) in walls
                ramp_top[j, i] = (x, y - 1, 0, 1) in ramps
                cliff_side[j, i] = (x - 1, y, 1, 0) in walls or (x, y, 1, 0) in walls
        CL = (cliff_top, ramp_top, cliff_side)
    plants = [t for t in tiles.values() if t["layer"] == "plant"]
    PL = None
    if "plant" in m.layers and plants:
        pcode = {t["id"]: i + 1 for i, t in enumerate(plants)}
        prow = m.layers["plant"]
        PL = (np.array([[pcode.get(prow[min(max(gy0 + j, 0), m.height - 1)][min(max(gx0 + i, 0), m.width - 1)], 0)
                         for i in range(gw)] for j in range(gh)], dtype=np.int16), plants)
    return {"P": P, "grounds": grounds, "g": _ground(G, X, Y, grounds, P, VX, VY, RF, CL, PL), "gx0": gx0, "gy0": gy0,
            "cells": (x0, y0, cw, ch)}


def render_frames(m, tiles: dict, pal: dict, frames, scale: int = 2, grid: bool = False, region=None,
                  objects: bool = True) -> list:
    """Pictures of the map (or of `region` = (x, y, w, h) in cells) at the given frames of the
    loop. The still part is worked out once and shared."""
    win = _window(m, tiles, pal, region)
    P, g, (x0, y0, cw, ch) = win["P"], win["g"], win["cells"]
    scale = max(1, min(int(scale), 4096 // (max(cw, ch) * C)))
    places = _place(m, tiles, win["grounds"], g, win["gx0"], win["gy0"], objects)
    sprites = _Sprites(tiles, P)
    shadow = _shadows(places, sprites, g["M"].shape)
    o, out = PAD * C, []
    for frame in frames:
        idx = _ground_frame(g, int(frame), P)
        _draw_things(idx, places, sprites, shadow, int(frame), P)
        img = Image.fromarray(P.rgba[idx[o:o + ch * C, o:o + cw * C]])  # uint8 h×w×4 is RGBA
        if scale > 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        if grid:
            line = (0, 0, 0, 255)
            for gx in range(1, cw):
                img.paste(line, (gx * C * scale, 0, gx * C * scale + 1, img.height))
            for gy in range(1, ch):
                img.paste(line, (0, gy * C * scale, img.width, gy * C * scale + 1))
        out.append(img)
    return out


def render(m, tiles: dict, pal: dict, scale: int = 2, grid: bool = False, region=None, objects: bool = True,
           frame: int = 0) -> Image.Image:
    return render_frames(m, tiles, pal, [frame], scale, grid, region, objects)[0]


def strip(images: list) -> Image.Image:
    """Frames stacked top to bottom in one picture."""
    out = Image.new("RGBA", (images[0].width, images[0].height * len(images)))
    for i, img in enumerate(images):
        out.paste(img, (0, i * img.height))
    return out


def instances(m, tiles: dict, pal: dict) -> list:
    """Every object and piece of decor on the map with where it stands in world pixels, so the
    game can draw, sort and animate them without repeating the placement rules here."""
    win = _window(m, tiles, pal, None)
    ox, oy = win["gx0"] * C, win["gy0"] * C
    return [{"tile": d["tile"], "variant": d["variant"], "flip": d["flip"], "x": d["fx"] + ox, "y": d["fy"] + oy,
             "phase": d["phase"], "wind": round(d["wind"], 4), "cell": list(d["cell"]), "decor": d["decor"],
             "shadow": d["shadow"]}
            for d in _place(m, tiles, win["grounds"], win["g"], win["gx0"], win["gy0"], True) if d["tile"]]


def sprite_frames(tile: dict, variant: int, pal: dict) -> list:
    """An object's sprite at every frame of its motion (one picture if it does not move)."""
    P = Palette(pal)
    base = P.sprite(tile["variants"][variant])
    motion = tile.get("motion", "none")
    count = 1 if motion == "none" else 2 if motion == "gust" else CYCLE
    return [P.image(moved(base, motion, f)) for f in range(count)]


FORMS = ("blades", "fronds", "tufts", "crust")   # how a plant stands: see clean_grow


def clean_grow(grow, pal: dict) -> dict:
    """How a plant grows, checked. height [low, high] px a stem stands; density stems per px
    of ground; colors root .. tip; under the colour of the ground in among the stems; spill px
    it reaches past its own cells (long grass leans out over its neighbours); bloom colours and
    blooming share for flowers or seed heads at the tips.
    form: blades - straight stems (grass, reeds); fronds - two leaves arching out from each root
    (fern, bracken); tufts - low and wide, a cushion (heather, gorse, thrift); crust - flat on
    the ground, unmoved by wind (lichen, moss). width 1..3 px a stem is wide; stiff - bends only
    a little in a gust (reeds); still - never moves."""
    if not isinstance(grow, dict):
        raise MapError("植被图块需要 grow：{height, density, colors, under, spill}")
    def color(c):
        if not isinstance(c, str) or c.startswith("_") or c not in pal:
            raise MapError(f"调色板里没有颜色 {c!r}")
        return c
    h = grow.get("height", [3, 6])
    if not (isinstance(h, list) and len(h) == 2 and all(isinstance(v, (int, float)) for v in h) and 1 <= h[0] <= h[1] <= 16):
        raise MapError("grow.height 应该是 [最矮, 最高]，1 到 16 像素")
    cols = grow.get("colors") or []
    if not (isinstance(cols, list) and 2 <= len(cols) <= 6):
        raise MapError("grow.colors 应该是 2 到 6 个颜色（从根到尖）")
    return {"height": [int(h[0]), int(h[1])], "density": float(min(max(float(grow.get("density", 0.2)), 0.01), 0.8)),
            "colors": [color(c) for c in cols], "under": color(grow.get("under", cols[0])),
            "spill": int(min(max(int(grow.get("spill", 2)), 0), 8)),
            "bloom": [color(c) for c in (grow.get("bloom") or [])][:6],
            "blooming": float(min(max(float(grow.get("blooming", 0.15 if grow.get("bloom") else 0)), 0), 1)),
            **_form(grow)}


def _form(grow) -> dict:
    form = grow.get("form", "blades")
    if form not in FORMS:
        raise MapError(f"grow.form 只能是 {' / '.join(FORMS)}")
    width = {"tufts": 3, "crust": 2}.get(form, 1)
    return {"form": form, "width": int(min(max(int(grow.get("width", width)), 1), 3)),
            "stiff": bool(grow.get("stiff", False)), "still": bool(grow.get("still", form == "crust"))}


def look_sample(tile: dict, tiles: dict, pal: dict, scale: int = 3, frame: int = 0) -> Image.Image:
    """A ground tile shown as a patch against its neighbour in rank, the way it will meet it
    on a map: on top of the next lower material, or - for the lowest - inside the next higher."""
    everything = {**tiles, tile["id"]: tile}
    if tile["layer"] == "plant":   # a patch of it on meadow
        n = 9
        m = mapkit.Map("sample", n, n)
        under = "grass" if "grass" in everything else next(t["id"] for t in everything.values() if t["layer"] == "ground")
        m.layers["ground"] = [[under] * n for _ in range(n)]
        m.layers["plant"] = [[tile["id"] if abs(xx - 4) + abs(yy - 4) <= 4 else None for xx in range(n)] for yy in range(n)]
        return render(m, everything, pal, scale=scale, frame=frame)
    grounds = sorted((t for t in everything.values() if t["layer"] == "ground"), key=lambda t: t.get("rank", 0))
    i = next(n for n, t in enumerate(grounds) if t["id"] == tile["id"])
    other = grounds[i - 1] if i > 0 else (grounds[1] if len(grounds) > 1 else None)
    n = 9
    m = mapkit.Map("sample", n, n)
    for yy in range(n):
        for xx in range(n):
            inside = abs(xx - 4) + abs(yy - 4) <= 4 and max(abs(xx - 4), abs(yy - 4)) <= 3
            m.layers["ground"][yy][xx] = tile["id"] if inside or other is None else other["id"]
    return render(m, everything, pal, scale=scale, frame=frame)
