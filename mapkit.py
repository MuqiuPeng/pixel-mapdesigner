"""Core of the map designer: the tileset, maps, edits, checks and rendering.

The web editor (server.py) and the MCP server (mcp_server.py) are both thin shells over this
module and work on the same files, so an edit made through one shows up in the other.

There are two kinds of tile. A *ground* tile is a material: it has no picture of its own, only
a `look` (colours, patches, specks, rim, cliff, shore, decor) from which render.py draws
continuous terrain. An *object* tile is a sprite of any size (up to 64 px) standing on a cell.

Sprite art is stored the way pixelkit's `from_grid` takes it - text rows plus a key of palette
names - so everything stays inside the project palette and can be redrawn after a palette
change. Recipes (`recipe_art`) run pixelkit's primitives and bake the result to that form.

Files:
    data/maps/tileset.json     tile definitions and their art
    data/maps/<name>.map.json  one map
    export/...                 what a game loads, written by `Project.export`
"""
from __future__ import annotations

import io
import json
import math
import os
import random
import re
import sys
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pixelkit as pk  # noqa: E402
from PIL import Image, ImageChops  # noqa: E402

CELL = 16
LAYERS = ("ground", "plant", "object")   # plant: what grows on the ground and can be cut, trodden, hidden in
PASS = ("walk", "swim", "block")
MAX_SIDE = 512
MAX_ART = 64
SHEET_WIDTH = 512
ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
NAME_RE = re.compile(r"^[A-Za-z0-9_\-一-鿿]{1,40}$")
ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MISSING = (255, 0, 255, 255)

# Speck shapes for `scatter`, by name, so recipes stay readable.
SHAPES = {
    "dot": [(0, 0)],
    "h2": [(0, 0), (1, 0)],
    "h3": [(0, 0), (1, 0), (2, 0)],
    "h4": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "v2": [(0, 0), (0, -1)],
    "l": [(0, 0), (1, 0), (0, 1)],
    "diag": [(0, 0), (1, 1)],
    "step": [(0, 0), (1, 0), (1, -1)],
    "zig": [(0, 0), (0, 1), (1, 1)],
}


def utf8_stdio() -> None:
    """Messages are Chinese; a piped stdout on Windows defaults to a codepage that has none."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


class MapError(ValueError):
    """A request that cannot be carried out; the message is meant for the person asking."""


# ---------------------------------------------------------------- tile art

def _color(pal: dict, name):
    if not isinstance(name, str) or name.startswith("_") or name not in pal:
        raise MapError(f"调色板里没有颜色 {name!r}")
    return pal[name]


def art_image(art: dict, pal: dict) -> Image.Image:
    """A sprite from `{"rows": [strings of equal length], "key": {char: palette name}}`."""
    try:
        rows, key = list(art["rows"]), dict(art["key"])
    except (KeyError, TypeError):
        raise MapError("图块的 art 需要 rows 和 key")
    if not rows or any(not isinstance(r, str) or len(r) != len(rows[0]) for r in rows):
        raise MapError("像素画的每一行都要一样长")
    if not (1 <= len(rows[0]) <= MAX_ART and len(rows) <= MAX_ART):
        raise MapError(f"像素画最大 {MAX_ART}×{MAX_ART}")
    for name in key.values():
        _color(pal, name)
    try:
        return pk.from_grid(rows, key, pal)
    except ValueError as e:
        raise MapError(str(e))


def image_art(img: Image.Image, pal: dict, snap: bool = False) -> dict:
    """The reverse of `art_image`. Every opaque pixel must be a palette colour; with `snap`
    stray colours go to the nearest one and soft alpha is cut at half."""
    img = img.convert("RGBA")
    if img.width > MAX_ART or img.height > MAX_ART:
        raise MapError(f"像素画最大 {MAX_ART}×{MAX_ART}，这张是 {img.width}×{img.height}")
    names = {}
    for name, value in pal.items():
        if not name.startswith("_"):
            names.setdefault(value, name)
    chars, rows = {}, []
    px = img.load()
    for y in range(img.height):
        row = ""
        for x in range(img.width):
            c = px[x, y]
            if c[3] == 0 or (snap and c[3] < 128):
                row += "."
                continue
            if c[3] != 255 and not snap:
                raise MapError(f"({x},{y}) 是半透明像素")
            c = c[:3] + (255,)
            name = names.get(c)
            if name is None:
                if not snap:
                    raise MapError("颜色 #%02x%02x%02x 不在调色板里" % c[:3])
                name = names[min(names, key=lambda v: sum((a - b) ** 2 for a, b in zip(v, c)))]
            if name not in chars:
                chars[name] = ALPHABET[len(chars)]
            row += chars[name]
        rows.append(row)
    return {"rows": rows, "key": {ch: name for name, ch in chars.items()}}


def recipe_art(recipe: dict, pal: dict) -> dict:
    """Draw a sprite with pixelkit's primitives and bake it to rows/key.

    `{"size": [w, h] (default 16×16), "fill": colour or null, "ops": [...], "outline": colour
    or null}` where an op is one of
      {"op": "rect", "x", "y", "w", "h", "color"}
      {"op": "px", "x", "y", "color"}
      {"op": "blob", "cx", "cy", "rx", "ry", "ramp": ramp name or [colours dark to light], "bands"?}
      {"op": "scatter", "seed", "count", "colors": [...], "shapes": [names in SHAPES], "wrap"?}
      {"op": "grid", "rows", "key", "x"?, "y"?}
    """
    try:
        fill = recipe.get("fill")
        w, h = recipe.get("size") or (CELL, CELL)
        if not (1 <= int(w) <= MAX_ART and 1 <= int(h) <= MAX_ART):
            raise MapError(f"像素画最大 {MAX_ART}×{MAX_ART}")
        c = pk.Canvas(int(w), int(h), _color(pal, fill) if fill else None)
        for op in recipe.get("ops", []):
            kind = op.get("op")
            if kind == "rect":
                c.rect(int(op["x"]), int(op["y"]), int(op["w"]), int(op["h"]), _color(pal, op["color"]))
            elif kind == "px":
                c.px(int(op["x"]), int(op["y"]), _color(pal, op["color"]))
            elif kind == "blob":
                r = op["ramp"]
                if isinstance(r, str):
                    if r not in pal["_ramps"]:
                        raise MapError(f"调色板里没有色阶 {r!r}")
                    colors = pk.ramp(pal, r)
                else:
                    colors = [_color(pal, n) for n in r]
                c.blob(float(op["cx"]), float(op["cy"]), float(op["rx"]), float(op["ry"]), colors,
                       bands=op.get("bands"))
            elif kind == "scatter":
                shapes = []
                for s in op.get("shapes", ["dot"]):
                    if s not in SHAPES:
                        raise MapError(f"没有形状 {s!r}，可用：{', '.join(SHAPES)}")
                    shapes.append(SHAPES[s])
                c.scatter(int(op.get("seed", 1)), int(op["count"]), [_color(pal, n) for n in op["colors"]],
                          shapes=shapes, wrap=bool(op.get("wrap", True)))
            elif kind == "grid":
                c.paste(pk.from_grid(list(op["rows"]), dict(op["key"]), pal), int(op.get("x", 0)), int(op.get("y", 0)))
            else:
                raise MapError(f"不认识的绘制操作 {kind!r}")
        img = c.img
        if recipe.get("outline"):
            img = pk.outline(img, _color(pal, recipe["outline"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        if isinstance(e, MapError):
            raise
        raise MapError(f"配方有误：{e!r}")
    return image_art(img, pal)


def art_op(op: str, art: dict, params: dict, pal: dict) -> dict:
    """A pixelkit transform applied to existing art."""
    img = art_image(art, pal)
    if op == "outline":
        img = pk.outline(img, _color(pal, params.get("color", "n0")), diagonal=bool(params.get("diagonal")))
    elif op == "mirror":
        img = pk.mirror(img)
    elif op == "shift":  # wraps around the edges
        img = ImageChops.offset(img, int(params.get("dx", 0)), int(params.get("dy", 0)))
    elif op == "swap":
        img = pk.swap(img, {_color(pal, a): _color(pal, b) for a, b in dict(params.get("map", {})).items()})
    else:
        raise MapError(f"不认识的图块操作 {op!r}")
    return image_art(img, pal)


def resolve_art(spec: dict, pal: dict) -> dict:
    """Art given either directly (rows/key) or as a recipe (fill/ops/outline)."""
    if not isinstance(spec, dict):
        raise MapError("art 应该是一个对象")
    if "rows" in spec:
        art = {"rows": list(spec["rows"]), "key": dict(spec.get("key", {}))}
        art_image(art, pal)
        if spec.get("outline"):
            art = art_op("outline", art, {"color": spec["outline"]}, pal)
        return art
    if "ops" in spec or "fill" in spec:
        return recipe_art(spec, pal)
    raise MapError("art 要么给 rows+key，要么给 fill/ops 配方")


def tile_preview(tile: dict, pal: dict, tiles: dict | None = None) -> Image.Image:
    """An enlarged picture for judging a tile. A ground tile is shown as a patch meeting its
    neighbour in rank; an object as each of its variants on a dark and a light backdrop, with
    the true-size sprites in the strip above."""
    if tile.get("layer") in ("ground", "plant"):
        import render as R
        return R.look_sample(tile, tiles or {}, pal)
    imgs = [art_image(a, pal) for a in tile["variants"]]
    f = max(2, min(8, 160 // max(max(i.width, i.height) for i in imgs)))
    gap, top = 6, max(i.height for i in imgs) + 6
    width = sum(i.width * f + gap for i in imgs) - gap
    out = Image.new("RGBA", (width * 2 + gap, top + max(i.height for i in imgs) * f), (20, 20, 24, 255))
    x = 3
    for i in imgs:
        out.alpha_composite(i, (x, 3))
        x += i.width + 3
    for n, base in enumerate([(58, 60, 70, 255), (200, 205, 190, 255)]):
        x = n * (width + gap)
        for i in imgs:
            panel = Image.new("RGBA", (i.width * f, i.height * f), base)
            panel.alpha_composite(pk.scale(i, f))
            out.alpha_composite(panel, (x, top))
            x += i.width * f + gap
    return out


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- maps

class Map:
    """Two layers of tile ids (None = empty): the ground, and what stands on it."""

    def __init__(self, name: str, width: int, height: int):
        self.name, self.width, self.height = name, width, height
        self.layers = {L: [[None] * width for _ in range(height)] for L in LAYERS}
        self.relief = None  # optional: how high each cell stands, 0 = sea
        self.ramps = []     # places a cliff can be climbed: [x, y, dx, dy], the edge from (x, y) to (x+dx, y+dy)
        self.scale = list(SCALE)   # [metres a cell is across, metres a relief unit stands]: what makes a slope steep
        self.biome = None   # optional: the biome of each cell (geo.BIOMES ids) - for the game: what lives there, the weather, the sound

    @classmethod
    def from_json(cls, d: dict) -> "Map":
        m = cls(d["name"], int(d["width"]), int(d["height"]))
        legend = d.get("legend", [])
        for L in LAYERS:
            rows = d.get(L) or []
            for y in range(min(m.height, len(rows))):
                for x in range(min(m.width, len(rows[y]))):
                    i = rows[y][x]
                    m.layers[L][y][x] = legend[i] if 0 <= i < len(legend) else None
        if d.get("relief"):
            m.relief = [[int(d["relief"][y][x]) if y < len(d["relief"]) and x < len(d["relief"][y]) else 0
                         for x in range(m.width)] for y in range(m.height)]
        m.ramps = [list(map(int, r)) for r in d.get("ramps", []) if len(r) == 4]
        if len(d.get("scale") or []) == 2:
            m.scale = [float(v) for v in d["scale"]]
        if d.get("biome"):
            legend, rows = d["biome"]["legend"], d["biome"]["cells"]
            m.biome = [[legend[rows[y][x]] if y < len(rows) and x < len(rows[y]) and 0 <= rows[y][x] < len(legend) else None
                        for x in range(m.width)] for y in range(m.height)]
        return m

    def to_json(self) -> dict:
        legend = sorted({t for L in LAYERS for row in self.layers[L] for t in row if t})
        at = {t: i for i, t in enumerate(legend)}
        d = {"name": self.name, "width": self.width, "height": self.height, "legend": legend}
        for L in LAYERS:
            d[L] = [[at[t] if t else -1 for t in row] for row in self.layers[L]]
        if self.relief:
            d["relief"] = [list(row) for row in self.relief]
        if self.ramps:
            d["ramps"] = [list(r) for r in self.ramps]
        if self.relief:
            d["scale"] = [round(v, 3) for v in self.scale]
        if self.biome:
            legend = sorted({b for row in self.biome for b in row if b})
            at = {b: i for i, b in enumerate(legend)}
            d["biome"] = {"legend": legend, "cells": [[at.get(b, -1) for b in row] for row in self.biome]}
        return d

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def used(self) -> dict:
        """tile id -> how many cells hold it."""
        n = {}
        for L in LAYERS:
            for row in self.layers[L]:
                for t in row:
                    if t:
                        n[t] = n.get(t, 0) + 1
        return n


def _map_text(m: Map) -> str:
    """JSON with one map row per line, so the file reads like the map and diffs by row."""
    d = m.to_json()

    def rows(L):
        return ",\n".join("  " + json.dumps(r, separators=(",", ":")) for r in d[L])

    head = ",\n".join(' "%s": %s' % (k, json.dumps(d[k], ensure_ascii=False)) for k in ("name", "width", "height", "legend"))
    tail = ",\n \"relief\": [\n%s\n ]" % rows("relief") if d.get("relief") else ""
    if d.get("ramps"):
        tail += ',\n "ramps": %s' % json.dumps(d["ramps"], separators=(",", ":"))
    if d.get("scale"):
        tail += ',\n "scale": %s' % json.dumps(d["scale"])
    if d.get("biome"):
        tail += ',\n "biome": {"legend": %s, "cells": [\n%s\n ]}' % (
            json.dumps(d["biome"]["legend"]), ",\n".join("  " + json.dumps(r, separators=(",", ":")) for r in d["biome"]["cells"]))
    plant = ",\n \"plant\": [\n%s\n ]" % rows("plant") if any(t >= 0 for row in d["plant"] for t in row) else ""
    return "{\n%s,\n \"ground\": [\n%s\n ]%s,\n \"object\": [\n%s\n ]%s\n}\n" % (
        head, rows("ground"), plant, rows("object"), tail)


def _check_size(width, height):
    if not (isinstance(width, int) and isinstance(height, int) and 1 <= width <= MAX_SIDE and 1 <= height <= MAX_SIDE):
        raise MapError(f"地图边长要在 1 到 {MAX_SIDE} 之间")


def _tile_for(tiles: dict, tile_id: str) -> dict:
    if tile_id not in tiles:
        raise MapError(f"没有图块 {tile_id!r}，现有：{', '.join(tiles) or '（空）'}")
    return tiles[tile_id]


def set_cells(m: Map, tiles: dict, cells: list) -> int:
    """Apply `[layer, x, y, tile id or None]` entries. Returns how many cells changed.
    Nothing is applied if any entry is invalid."""
    checked = []
    for c in cells:
        try:
            layer, x, y, t = c[0], int(c[1]), int(c[2]), c[3]
        except (TypeError, ValueError, IndexError):
            raise MapError(f"格子条目应为 [层, x, y, 图块]：{c!r}")
        if layer not in LAYERS:
            raise MapError(f"没有 {layer!r} 层，只有 {' / '.join(LAYERS)}")
        if not m.inside(x, y):
            raise MapError(f"({x},{y}) 在地图 {m.width}×{m.height} 之外")
        if t is not None and _tile_for(tiles, t)["layer"] != layer:
            raise MapError(f"图块 {t!r} 属于 {tiles[t]['layer']} 层，不能放到 {layer} 层")
        checked.append((layer, x, y, t))
    changed = 0
    for layer, x, y, t in checked:
        if m.layers[layer][y][x] != t:
            m.layers[layer][y][x] = t
            changed += 1
    return changed


def _clip(m: Map, x, y, w, h):
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(m.width, int(x) + int(w)), min(m.height, int(y) + int(h))
    return [(xx, yy) for yy in range(y0, y1) for xx in range(x0, x1)]


def fill_rect(m: Map, tiles: dict, tile_id, x, y, w, h, layer=None) -> int:
    """Fill a rectangle with a tile; `tile_id` None erases `layer`. Clipped to the map."""
    layer = _tile_for(tiles, tile_id)["layer"] if tile_id is not None else layer
    return set_cells(m, tiles, [[layer, xx, yy, tile_id] for xx, yy in _clip(m, x, y, w, h)])


def line_cells(x0, y0, x1, y1) -> list:
    """Bresenham."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err, out = dx + dy, []
    while True:
        out.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return out
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_line(m: Map, tiles: dict, tile_id, x0, y0, x1, y1, width=1) -> int:
    layer = _tile_for(tiles, tile_id)["layer"]
    width = max(1, int(width))
    off = range(-((width - 1) // 2), width // 2 + 1)
    seen = {(x + ox, y + oy) for x, y in line_cells(x0, y0, x1, y1) for ox in off for oy in off}
    return set_cells(m, tiles, [[layer, x, y, tile_id] for x, y in sorted(seen) if m.inside(x, y)])


def flood_fill(m: Map, tiles: dict, tile_id, x, y) -> int:
    """Replace the 4-connected region of equal cells around (x, y) on the tile's layer."""
    layer = _tile_for(tiles, tile_id)["layer"]
    x, y = int(x), int(y)
    if not m.inside(x, y):
        raise MapError(f"({x},{y}) 在地图之外")
    grid = m.layers[layer]
    old = grid[y][x]
    if old == tile_id:
        return 0
    todo, n = deque([(x, y)]), 0
    grid[y][x] = tile_id
    while todo:
        cx, cy = todo.popleft()
        n += 1
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if m.inside(nx, ny) and grid[ny][nx] == old:
                grid[ny][nx] = tile_id
                todo.append((nx, ny))
    return n


def footprint_cells(tile: dict, x: int, y: int) -> list:
    """The cells an object covers on the ground. It stands on (x, y) - the middle of its
    front edge - and a footprint of [w, h] reaches w cells across and h cells back (up)."""
    w, h = tile.get("footprint") or (1, 1)
    left = x - (w - 1) // 2
    return [(xx, yy) for yy in range(y - h + 1, y + 1) for xx in range(left, left + w)]


TIER = 8          # relief units to a step of the land
SCALE = (12.4, 3.3)   # unless a map says: a cell is 12.4 m across, a relief unit 3.3 m high
CLIFF_DEG = 40.0  # a slope steeper than this, measured on the ground, is a cliff; anything gentler can be walked


def gradient(m: Map) -> list:
    """How steep the land is at each cell, in degrees: the slope of its relief, from the
    neighbours on either side, at the map's scale."""
    rel, W, H = m.relief, m.width, m.height
    cell_m, unit_m = m.scale

    def d(a, b, c):   # the change across a cell, from what is on each side
        return (c - a) / 2 if a is not None and c is not None else (c - b if c is not None else b - a if a is not None else 0)

    out = []
    for y in range(H):
        row = []
        for x in range(W):
            r = rel[y][x]
            gx = d(rel[y][x - 1] if x > 0 else None, r, rel[y][x + 1] if x + 1 < W else None)
            gy = d(rel[y - 1][x] if y > 0 else None, r, rel[y + 1][x] if y + 1 < H else None)
            row.append(math.degrees(math.atan(math.hypot(gx, gy) * unit_m / cell_m)))
        out.append(row)
    return out


def tiers(m: Map) -> list:
    """Which step of the land each cell is on (None for a map without heights). A lone cell on a
    step of its own goes with its neighbours."""
    if not m.relief:
        return None
    t = [[r // TIER for r in row] for row in m.relief]
    for _ in range(2):
        out = [row[:] for row in t]
        for y in range(m.height):
            for x in range(m.width):
                near = [t[ny][nx] for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if m.inside(nx, ny)]
                if near and all(n != t[y][x] for n in near):
                    out[y][x] = sorted(near)[len(near) // 2]
        t = out
    return t


def cliffs(m: Map, tiles: dict) -> set:
    """Edges between cells that are cliffs: a step of the land where the ground itself is nearly
    vertical (steeper than CLIFF_DEG, from the local gradient), between two pieces of dry ground,
    and not a ramp. Each edge is (x, y, dx, dy) with (dx, dy) = (1, 0) or (0, 1). Most steps are
    slopes one can walk up, even steep ones; only a real face is a cliff."""
    t = tiers(m)
    if t is None:
        return set()
    rel, ground = m.relief, m.layers["ground"]

    def dry(x, y):
        g = tiles.get(ground[y][x]) if ground[y][x] else None
        return bool(g) and g["layer"] == "ground" and g.get("pass") != "swim" and not g["look"].get("flows")

    grade = gradient(m)
    cell_m, unit_m = m.scale
    ramps = {tuple(r) for r in m.ramps}
    out = set()
    for y in range(m.height):
        for x in range(m.width):
            for dx, dy in ((1, 0), (0, 1)):
                nx, ny = x + dx, y + dy
                if not m.inside(nx, ny) or t[y][x] == t[ny][nx] or (x, y, dx, dy) in ramps:
                    continue
                if not (dry(x, y) and dry(nx, ny)):
                    continue
                across = math.degrees(math.atan(abs(rel[y][x] - rel[ny][nx]) * unit_m / cell_m))
                if min(across, (grade[y][x] + grade[ny][nx]) / 2) >= CLIFF_DEG:
                    out.add((x, y, dx, dy))
    return out


def blocked_edge(edges: set, x, y, nx, ny) -> bool:
    """Whether stepping from (x, y) to the neighbouring (nx, ny) crosses one of `edges`."""
    a, b = ((x, y), (nx, ny)) if (x, y) <= (nx, ny) else ((nx, ny), (x, y))
    if a[1] == b[1]:
        a, b = ((x, y), (nx, ny)) if x < nx else ((nx, ny), (x, y))
    else:
        a, b = ((x, y), (nx, ny)) if y < ny else ((nx, ny), (x, y))
    return (a[0], a[1], b[0] - a[0], b[1] - a[1]) in edges


def land_regions(m: Map, tiles: dict):
    """Walkable land split into 4-connected pieces: (region id per cell or -1,
    [(size, first cell)] per region, id of the largest or -1). A cell is open when its ground
    can be walked on and nothing blocking stands on it."""
    ground, objects = m.layers["ground"], m.layers["object"]

    blocked = set()
    for y in range(m.height):
        for x in range(m.width):
            t = tiles.get(objects[y][x]) if objects[y][x] else None
            if t and t.get("pass") == "block":
                blocked.update(footprint_cells(t, x, y))

    walls = cliffs(m, tiles)

    def open_cell(x, y):
        g = tiles.get(ground[y][x], {}).get("pass", "block") if ground[y][x] else "block"
        return g == "walk" and (x, y) not in blocked

    region = [[-1] * m.width for _ in range(m.height)]
    sizes = []
    for y in range(m.height):
        for x in range(m.width):
            if region[y][x] != -1 or not open_cell(x, y):
                continue
            rid, n, todo = len(sizes), 0, deque([(x, y)])
            region[y][x] = rid
            while todo:
                cx, cy = todo.popleft()
                n += 1
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if m.inside(nx, ny) and region[ny][nx] == -1 and open_cell(nx, ny) \
                            and not (walls and blocked_edge(walls, cx, cy, nx, ny)):
                        region[ny][nx] = rid
                        todo.append((nx, ny))
            sizes.append((n, (x, y)))
    main = max(range(len(sizes)), key=lambda i: sizes[i][0]) if sizes else -1
    return region, sizes, main


def scatter(m: Map, tiles: dict, tile_id, count, seed=1, on=None, region=None, spacing=0, reachable=False) -> list:
    """Place up to `count` copies of an object tile on free cells, seeded. `on`: ground tile
    ids it may stand on (default: any walkable ground). `spacing`: minimum distance, in
    cells, to any other object. `reachable`: only on the largest piece of walkable land.
    Returns the cells used."""
    if _tile_for(tiles, tile_id)["layer"] != "object":
        raise MapError("scatter 只用于 object 层的图块")
    for g in on or []:
        _tile_for(tiles, g)
    x, y, w, h = region or (0, 0, m.width, m.height)
    ground, objects = m.layers["ground"], m.layers["object"]

    def allowed(g):
        if g is None or g not in tiles:
            return False
        return g in on if on else tiles[g].get("pass") == "walk"

    cells = [(cx, cy) for cx, cy in _clip(m, x, y, w, h) if objects[cy][cx] is None and allowed(ground[cy][cx])]
    if reachable:
        land, _, main = land_regions(m, tiles)
        cells = [(cx, cy) for cx, cy in cells if land[cy][cx] == main]
    random.Random(f"{seed}:{tile_id}").shuffle(cells)
    spacing = max(0, int(spacing))
    placed = []
    taken = set()  # ground already under something, counting whole footprints
    for yy in range(m.height):
        for xx in range(m.width):
            if objects[yy][xx] in tiles:
                taken.update(footprint_cells(tiles[objects[yy][xx]], xx, yy))
    for cx, cy in cells:
        if len(placed) >= int(count):
            break
        mine = footprint_cells(tiles[tile_id], cx, cy)
        if any(not m.inside(xx, yy) or (xx, yy) in taken or objects[yy][xx] is not None or not allowed(ground[yy][xx])
               for xx, yy in mine):
            continue
        if spacing and any(objects[yy][xx] is not None or (xx, yy) in taken
                           for xx, yy in _clip(m, cx - spacing, cy - spacing, 2 * spacing + 1, 2 * spacing + 1)):
            continue
        objects[cy][cx] = tile_id
        taken.update(mine)
        placed.append((cx, cy))
    return placed


def resize(m: Map, width, height, dx=0, dy=0, fill=None, tiles=None) -> Map:
    """A new map of the given size with the old content moved by (dx, dy); uncovered ground
    takes `fill`."""
    _check_size(width, height)
    if fill is not None and _tile_for(tiles or {}, fill)["layer"] != "ground":
        raise MapError("填充要用 ground 层的图块")
    out = Map(m.name, width, height)
    if fill is not None:
        out.layers["ground"] = [[fill] * width for _ in range(height)]
    for L in LAYERS:
        for y in range(m.height):
            for x in range(m.width):
                if out.inside(x + dx, y + dy):
                    out.layers[L][y + dy][x + dx] = m.layers[L][y][x]
    if m.relief:
        out.relief = [[0] * width for _ in range(height)]
        for y in range(m.height):
            for x in range(m.width):
                if out.inside(x + dx, y + dy):
                    out.relief[y + dy][x + dx] = m.relief[y][x]
    out.ramps = [[x + dx, y + dy, ex, ey] for x, y, ex, ey in m.ramps
                 if out.inside(x + dx, y + dy) and out.inside(x + dx + ex, y + dy + ey)]
    out.scale = list(m.scale)
    if m.biome:
        out.biome = [[None] * width for _ in range(height)]
        for y in range(m.height):
            for x in range(m.width):
                if out.inside(x + dx, y + dy):
                    out.biome[y + dy][x + dx] = m.biome[y][x]
    return out


def validate(m: Map, tiles: dict) -> dict:
    """Structural checks. Problems: unknown tiles, holes in the ground, objects that cannot
    be reached from the largest piece of walkable land. Notes: the land being in pieces."""
    problems = []
    used = m.used()
    for t in sorted(used):
        if t not in tiles:
            problems.append(f"图块 {t!r} 不在图块集中（{used[t]} 格）")
    ground, objects = m.layers["ground"], m.layers["object"]
    holes = [(x, y) for y in range(m.height) for x in range(m.width) if ground[y][x] is None]
    if holes:
        problems.append(f"{len(holes)} 格没有地面，例如 {holes[:3]}")

    region, sizes, main = land_regions(m, tiles)
    notes = []  # worth knowing, not wrong: islands are fine as long as nothing needed is on them
    if not sizes:
        problems.append("没有任何可以行走的格子")
    elif len(sizes) > 1:
        rest = sorted((s for i, s in enumerate(sizes) if i != main), reverse=True)
        notes.append("可行走的陆地被分成 %d 块：最大 %d 格，其余 %s" % (
            len(sizes), sizes[main][0], "、".join(f"{n} 格（起于 {at}）" for n, at in rest[:5])
            + ("……" if len(rest) > 5 else "")))
    cut_off = []
    for y in range(m.height):
        for x in range(m.width):
            if objects[y][x] is None or main < 0:
                continue
            near = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            if not any(m.inside(nx, ny) and region[ny][nx] == main for nx, ny in near):
                cut_off.append((objects[y][x], x, y))
    if cut_off:
        problems.append("%d 个物件从最大的陆地走不到，例如 %s" % (
            len(cut_off), "、".join(f"{t}@({x},{y})" for t, x, y in cut_off[:5])))
    return {"ok": not problems, "problems": problems, "notes": notes,
            "stats": {"size": [m.width, m.height], "tiles": used,
                      "walkable_regions": len(sizes), "largest_region": sizes[main][0] if sizes else 0}}


def tile_chars(tiles: dict) -> dict:
    """A stable character per tile for text views: ground lowercase first, objects uppercase."""
    out = {}
    for layer, pool in (("ground", "abcdefghijklmnopqrstuvwxyz0123456789"), ("object", "ABCDEFGHIJKLMNOPQRSTUVWXYZ@#$%&*+="),
                        ("plant", "~^\"',;:`")):
        ids = [t for t in tiles if tiles[t]["layer"] == layer]
        for i, t in enumerate(ids):
            out[t] = pool[i] if i < len(pool) else "?"
    return out


def text_view(m: Map, tiles: dict, region=None, layer=None) -> str:
    """The map as text. One character per cell; an object hides the ground under it unless
    a single `layer` is asked for. '.' is empty, '!' a tile missing from the tileset."""
    chars = tile_chars(tiles)
    x, y, w, h = region or (0, 0, m.width, m.height)
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(m.width, x0 + int(w)), min(m.height, y0 + int(h))
    lines, seen = [], set()
    for yy in range(y0, y1):
        row = ""
        for xx in range(x0, x1):
            t = None
            for L in (("object", "plant", "ground") if layer is None else (layer,)):
                t = m.layers[L][yy][xx]
                if t:
                    break
            if t:
                seen.add(t)
            row += chars.get(t, "!") if t else "."
        lines.append(f"{yy:3d} {row}")
    legend = "  ".join(f"{chars.get(t, '!')}={t}" for t in sorted(seen, key=lambda t: (tiles.get(t, {}).get("layer", ""), t)))
    head = f"{m.name}  {m.width}×{m.height}  显示 x {x0}..{x1 - 1}, y {y0}..{y1 - 1}（每行开头是 y）"
    ruler = "    " + "".join(str((xx // 10) % 10) if xx % 10 == 0 else " " for xx in range(x0, x1))
    ruler2 = "    " + "".join(str(xx % 10) for xx in range(x0, x1))
    return "\n".join([head, ruler, ruler2, *lines, "图例: " + (legend or "（空）")])


def render(m: Map, tiles: dict, pal: dict, scale: int = 2, grid: bool = False, region=None,
           objects: bool = True, frame: int = 0) -> Image.Image:
    """The map as the game shows it at one frame of the animation loop; see render.py. The
    scale drops if the image would pass 4096 px."""
    import render as R
    return R.render(m, tiles, pal, scale, grid, region, objects, frame)


# ---------------------------------------------------------------- the project on disk

class Project:
    """Where the tileset and maps live. Every change is read-modify-write under a lock file,
    because the editor and the MCP server are separate processes."""

    def __init__(self, data_dir=None, palette_path=None, export_dir=None):
        self.data = Path(data_dir or os.environ.get("MAPDESIGNER_DATA") or ROOT / "data" / "maps")
        self.palette_path = Path(palette_path or os.environ.get("MAPDESIGNER_PALETTE") or ROOT / "palette.json")
        self.export_dir = Path(export_dir or os.environ.get("MAPDESIGNER_EXPORT") or ROOT / "export")
        self.tileset_path = self.data / "tileset.json"

    # -- plumbing

    @contextmanager
    def lock(self):
        self.data.mkdir(parents=True, exist_ok=True)
        path = self.data / ".lock"
        deadline = time.time() + 5
        while True:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except OSError:
                try:
                    if time.time() - path.stat().st_mtime > 10:  # left behind by a crash
                        path.unlink()
                        continue
                except OSError:
                    pass
                if time.time() > deadline:
                    raise MapError("地图数据正被另一个进程占用，稍后再试")
                time.sleep(0.02)
        try:
            yield
        finally:
            os.close(fd)
            try:
                path.unlink()
            except OSError:
                pass

    @staticmethod
    def _write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        for _ in range(10):  # a reader may hold the file open for a moment on Windows
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.03)
        os.replace(tmp, path)

    @staticmethod
    def _rev(path: Path):
        try:
            return str(path.stat().st_mtime_ns)
        except OSError:
            return None

    def palette(self) -> dict:
        return pk.load_palette(self.palette_path)

    def palette_json(self) -> dict:
        return json.loads(self.palette_path.read_text(encoding="utf-8"))

    # -- tileset

    def tileset(self) -> dict:
        if not self.tileset_path.exists():
            return {"cell": CELL, "tiles": []}
        return json.loads(self.tileset_path.read_text(encoding="utf-8"))

    def tiles(self) -> dict:
        return {t["id"]: t for t in self.tileset()["tiles"]}

    def tileset_rev(self):
        return self._rev(self.tileset_path)

    def _save_tileset(self, ts: dict) -> None:
        self._write(self.tileset_path, json.dumps(ts, ensure_ascii=False, indent=1) + "\n")

    def save_tile(self, tile: dict) -> dict:
        """Add a tile or replace the one with the same id; fields left out keep their old
        value. A ground tile needs `look`; an object needs `art` (one sprite) or `variants`
        (several, picked per cell). Returns the stored tile and what pixelkit's check says
        about its sprites (an empty list is clean)."""
        import render as R
        pal = self.palette()
        tid = tile.get("id")
        if not isinstance(tid, str) or not ID_RE.match(tid):
            raise MapError("图块 id 要用小写字母开头，只含小写字母、数字、下划线（最长 32）")
        given = {k: v for k, v in tile.items() if v is not None}
        with self.lock():
            ts = self.tileset()
            tiles = {t["id"]: t for t in ts["tiles"]}
            old = tiles.get(tid)
            merged = {**(old or {}), **given}
            layer, pas = merged.get("layer", "ground"), merged.get("pass", "walk")
            if layer not in LAYERS:
                raise MapError(f"layer 只能是 {' / '.join(LAYERS)}")
            if pas not in PASS:
                raise MapError(f"pass 只能是 {' / '.join(PASS)}")
            tags = merged.get("tags", [])
            if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
                raise MapError("tags 应该是字符串列表")
            if old and old["layer"] != layer:
                users = [m["name"] for m in self.list_maps() if tid in self.load_map(m["name"]).used()]
                if users:
                    raise MapError(f"图块 {tid!r} 已用在地图 {', '.join(users)} 里，不能换层")
            stored = {"id": tid, "name": str(merged.get("name") or tid), "layer": layer, "pass": pas,
                      "tags": [t.strip() for t in tags if t.strip()]}
            problems = []
            if layer == "plant":
                stored["grow"] = R.clean_grow(merged.get("grow"), pal)
            elif layer == "ground":
                ranks = [t.get("rank", 0) for t in ts["tiles"] if t["layer"] == "ground" and t["id"] != tid]
                stored["rank"] = int(merged["rank"]) if "rank" in merged else max(ranks, default=-1) + 1
                stored["look"] = R.clean_look(merged.get("look"), pal, tiles)
            else:
                arts = given.get("variants") or ([given["art"]] if "art" in given else merged.get("variants"))
                if not arts or not isinstance(arts, list) or len(arts) > 8:
                    raise MapError("物件图块需要 art（一张像素画）或 variants（1 到 8 张）")
                stored["variants"] = [resolve_art(a, pal) for a in arts]
                stored["shadow"] = bool(merged.get("shadow", False))
                stored["flip"] = bool(merged.get("flip", False))
                stored["jitter"] = max(0, min(6, int(merged.get("jitter", 0))))
                fp = merged.get("footprint") or [1, 1]
                if not (isinstance(fp, (list, tuple)) and len(fp) == 2 and all(isinstance(v, int) and 1 <= v <= 8 for v in fp)):
                    raise MapError("footprint 应该是 [宽, 深]，各 1 到 8 格")
                stored["footprint"] = list(fp)
                stored["motion"] = merged.get("motion") or "none"
                if stored["motion"] not in R.MOTIONS:
                    raise MapError(f"motion 只能是 {' / '.join(R.MOTIONS)}")
                for a in stored["variants"]:
                    problems += pk.check(art_image(a, pal), pal, max_colors=24)
            if old:
                ts["tiles"][ts["tiles"].index(old)] = stored
            else:
                ts["tiles"].append(stored)
            self._save_tileset(ts)
        return {"tile": stored, "problems": problems}

    def delete_tile(self, tid: str, force: bool = False) -> dict:
        """Remove a tile. If maps still use it this refuses, unless `force`, which also
        clears those cells."""
        with self.lock():
            ts = self.tileset()
            old = next((t for t in ts["tiles"] if t["id"] == tid), None)
            if old is None:
                raise MapError(f"没有图块 {tid!r}")
            cleared = {}
            for info in self.list_maps():
                m = self.load_map(info["name"])
                n = m.used().get(tid, 0)
                if n:
                    cleared[m.name] = n
            decor = [t["id"] for t in ts["tiles"] if t["layer"] == "ground" and tid != t["id"]
                     and any(d["tile"] == tid for d in t["look"]["decor"])]
            if decor and not force:
                raise MapError(f"图块 {tid!r} 是 {', '.join(decor)} 的点缀（decor）")
            for t in ts["tiles"]:
                if t["id"] in decor:
                    t["look"]["decor"] = [d for d in t["look"]["decor"] if d["tile"] != tid]
            if cleared and not force:
                raise MapError("图块 %r 还在用：%s" % (
                    tid, "、".join(f"{k}（{v} 格）" for k, v in cleared.items())))
            for name in cleared:
                m = self.load_map(name)
                for L in LAYERS:
                    m.layers[L] = [[None if t == tid else t for t in row] for row in m.layers[L]]
                self._save_map(m)
            ts["tiles"].remove(old)
            self._save_tileset(ts)
        return {"deleted": tid, "cleared": cleared}

    def export(self) -> dict:
        """Write what a game needs to show a map *moving*, under export/:

        tiles.png / tiles.json   every object sprite at every frame of its motion, and where
                                 each frame is on the sheet; loop length and speed
        maps/<name>.ground.png   the still ground
        maps/<name>.ground.N.png for each later frame N, only the pixels that differ from the
                                 still ground (the rest transparent): draw it over the ground
        maps/<name>.things.json  every object and piece of decor with its place in world
                                 pixels, variant, mirroring and phase, sorted back to front
        maps/<name>.plants.json  what grows in each cell (grass, short or long): it is drawn in
                                 the ground pictures, and the game keeps it per cell - cut, trodden,
                                 grown back
        """
        import render as R
        pal, ts = self.palette(), self.tileset()
        if not ts["tiles"]:
            raise MapError("图块集是空的")
        # Small motions (sway, flicker, bob) have CYCLE frames: show frame (now + phase) % cycle.
        # A `gust` object has two (standing, leaning): it leans while
        # ((its wind) - now / frames) mod 1 is under wind.width - that is, while a gust is on it.
        index = {"cell": CELL, "frames": R.FRAMES, "cycle": R.CYCLE, "fps": R.FPS,
                 "wind": {"direction": list(R.WIND), "span": R.GUST_SPAN, "width": R.GUST_WIDTH, "lean_above": 0.25},
                 "grounds": {}, "plants": {}, "objects": {}}
        placed, x, y, shelf = [], 0, 0, 0
        for t in ts["tiles"]:
            if t["layer"] == "ground":
                index["grounds"][t["id"]] = {"rank": t["rank"], "pass": t["pass"], "tags": t["tags"]}
                continue
            if t["layer"] == "plant":    # drawn into the ground pictures; the game keeps its state per cell
                index["plants"][t["id"]] = {"pass": t["pass"], "tags": t["tags"], "grow": t["grow"]}
                continue
            variants = []
            for vi in range(len(t["variants"])):
                rects = []
                for img in R.sprite_frames(t, vi, pal):
                    if x + img.width > SHEET_WIDTH:
                        x, y, shelf = 0, y + shelf + 1, 0
                    placed.append((img, x, y))
                    rects.append([x, y, img.width, img.height])
                    x, shelf = x + img.width + 1, max(shelf, img.height)
                variants.append(rects)
            index["objects"][t["id"]] = {"variants": variants, "motion": t.get("motion", "none"), "pass": t["pass"], "tags": t["tags"],
                                         "shadow": t["shadow"], "flip": t["flip"], "jitter": t["jitter"],
                                         "footprint": t["footprint"]}
        sheet = pk.blank(SHEET_WIDTH, max(1, y + shelf))
        for img, px, py in placed:
            sheet.alpha_composite(img, (px, py))
        png, js = self.export_dir / "tiles.png", self.export_dir / "tiles.json"
        pk.save(sheet, png)
        self._write(js, json.dumps(index, ensure_ascii=False, indent=1) + "\n")
        tiles, maps = self.tiles(), []
        for info in self.list_maps():
            m, stem = self.load_map(info["name"]), self.export_dir / "maps" / info["name"]
            frames = R.render_frames(m, tiles, pal, range(R.FRAMES), scale=1, objects=False)
            pk.save(frames[0], f"{stem}.ground.png")
            still, moving = frames[0].tobytes(), 0
            for n, img in enumerate(frames[1:], start=1):
                same = ImageChops.difference(img, frames[0]).convert("L").point(lambda v: 255 if v == 0 else 0)
                over = img.copy()
                over.paste((0, 0, 0, 0), mask=same)
                moving += img.tobytes() != still
                pk.save(over, f"{stem}.ground.{n}.png")
            things = R.instances(m, tiles, pal)
            plants = {"legend": sorted({p for row in m.layers["plant"] for p in row if p}), "cells": None}
            at = {p: i for i, p in enumerate(plants["legend"])}
            plants["cells"] = [[at[p] if p else -1 for p in row] for row in m.layers["plant"]]
            self._write(Path(f"{stem}.plants.json"), json.dumps(plants, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._write(Path(f"{stem}.things.json"), json.dumps(things, ensure_ascii=False, separators=(",", ":")) + "\n")
            maps.append({"name": info["name"], "ground": f"{stem}.ground.png", "moving_frames": moving, "things": len(things)})
        return {"png": str(png), "json": str(js), "objects": len(index["objects"]), "maps": maps,
                "problems": pk.check(sheet, pal, max_colors=len(pal))}

    # -- maps

    def _map_path(self, name: str) -> Path:
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise MapError("地图名只能含字母、数字、汉字、下划线和连字符（最长 40）")
        return self.data / f"{name}.map.json"

    def list_maps(self) -> list:
        out = []
        for p in sorted(self.data.glob("*.map.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                out.append({"name": d["name"], "width": d["width"], "height": d["height"], "rev": self._rev(p)})
            except (OSError, ValueError, KeyError):
                continue
        return out

    def map_rev(self, name: str):
        return self._rev(self._map_path(name))

    def load_map(self, name: str) -> Map:
        p = self._map_path(name)
        if not p.exists():
            have = ", ".join(m["name"] for m in self.list_maps()) or "（还没有地图）"
            raise MapError(f"没有地图 {name!r}，现有：{have}")
        return Map.from_json(json.loads(p.read_text(encoding="utf-8")))

    def _save_map(self, m: Map) -> None:
        self._write(self._map_path(m.name), _map_text(m))

    def create_map(self, name: str, width: int, height: int, fill=None) -> Map:
        _check_size(width, height)
        with self.lock():
            if self._map_path(name).exists():
                raise MapError(f"已经有地图 {name!r}")
            m = Map(name, width, height)
            if fill:
                fill_rect(m, self.tiles(), fill, 0, 0, width, height)
            self._save_map(m)
        return m

    def edit(self, name: str, fn, base_rev=None) -> dict:
        """Load a map, let `fn(map, tiles)` change it (or return a replacement Map), save.
        `stale` says the file had changed since `base_rev`, i.e. the caller's copy is old."""
        with self.lock():
            before = self.map_rev(name)
            m = self.load_map(name)
            result = fn(m, self.tiles())
            if isinstance(result, Map):
                m, result = result, None
            self._save_map(m)
            return {"result": result, "rev": self.map_rev(name), "stale": base_rev is not None and base_rev != before,
                    "map": m}

    def trash_map(self, name: str) -> str:
        """Move a map into .trash/ rather than deleting it."""
        with self.lock():
            p = self._map_path(name)
            if not p.exists():
                raise MapError(f"没有地图 {name!r}")
            dest = self.data / ".trash" / f"{name}.{time.strftime('%Y%m%d-%H%M%S')}.map.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(p, dest)
        return str(dest)

    def revs(self) -> dict:
        return {"tileset": self.tileset_rev(), "maps": {m["name"]: m["rev"] for m in self.list_maps()}}
