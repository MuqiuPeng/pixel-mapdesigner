"""Small toolkit for making pixel art from code. Pillow only.

Everything here keeps pixels hard: no resampling other than integer nearest-neighbour, no
partial alpha. Import it from an art source script:

    import sys; sys.path.insert(0, r"<project>/.claude/skills/pixel-art/scripts")
    from pixelkit import *
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image

CLEAR = (0, 0, 0, 0)


# ---------------------------------------------------------------- palette

def _rgba(hex_or_tuple):
    if isinstance(hex_or_tuple, (tuple, list)):
        t = tuple(hex_or_tuple)
        return t if len(t) == 4 else t + (255,)
    h = hex_or_tuple.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)


def find_project(start: Path | None = None) -> Path:
    """Nearest ancestor holding art/palette.json (or the cwd if none)."""
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "palette.json").exists() or (p / "art" / "palette.json").exists():
            return p
    return here


def load_palette(path: str | Path | None = None) -> dict:
    """{name: (r, g, b, a)}. The file may also define "ramps": {name: [colour names]}."""
    root = find_project()
    path = Path(path) if path else (root / "palette.json" if (root / "palette.json").exists() else root / "art" / "palette.json")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pal = {name: _rgba(value) for name, value in data["colors"].items()}
    pal["_ramps"] = data.get("ramps", {})
    return pal


def ramp(pal: dict, name: str) -> list:
    """Colours of a named ramp, dark to light."""
    return [pal[c] for c in pal["_ramps"][name]]


# ---------------------------------------------------------------- making images

def blank(w: int, h: int) -> Image.Image:
    return Image.new("RGBA", (w, h), CLEAR)


def from_grid(rows: list[str], key: dict, pal: dict) -> Image.Image:
    """Build a sprite from text rows. `key` maps a character to a palette name; '.' and ' '
    are transparent. Rows of unequal width are an error, on purpose."""
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        lines = "\n".join(f"  row {i:2d}: {len(r):2d} |{r}|" for i, r in enumerate(rows))
        raise ValueError("rows have different widths:\n" + lines)
    img = blank(len(rows[0]), len(rows))
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in ". ":
                continue
            if ch not in key:
                raise ValueError(f"row {y}: character {ch!r} is not in the key")
            img.putpixel((x, y), pal[key[ch]])
    return img


class Canvas:
    """A drawing surface for primitives. Coordinates are pixels, origin top-left."""

    def __init__(self, w: int, h: int, fill=None):
        self.img = Image.new("RGBA", (w, h), fill or CLEAR)
        self.w, self.h = w, h

    def px(self, x: int, y: int, color) -> "Canvas":
        if 0 <= x < self.w and 0 <= y < self.h:
            self.img.putpixel((x, y), color)
        return self

    def rect(self, x: int, y: int, w: int, h: int, color) -> "Canvas":
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                self.px(xx, yy, color)
        return self

    def blob(self, cx: float, cy: float, rx: float, ry: float, colors: list,
             light=(-0.6, -0.8), bands=None) -> "Canvas":
        """A filled ellipse shaded in flat bands from `colors` (dark to light), lit from
        `light` (a direction; the default is top-left). Bands, not a gradient: each pixel
        gets exactly one ramp colour."""
        n = len(colors)
        bands = bands or [i / n for i in range(1, n)]  # thresholds between bands, 0..1
        for y in range(self.h):
            for x in range(self.w):
                dx, dy = (x + 0.5 - cx) / rx, (y + 0.5 - cy) / ry
                if dx * dx + dy * dy > 1.0:
                    continue
                # 0 = facing away from the light, 1 = facing it
                lit = 0.5 + 0.5 * (dx * light[0] + dy * light[1])
                lit = min(0.999, max(0.0, lit))
                idx = sum(1 for t in bands if lit >= t)
                self.px(x, y, colors[idx])
        return self

    def scatter(self, seed: int, count: int, colors: list, shapes=((0, 0),), wrap=True,
                only_on=None) -> "Canvas":
        """Seeded specks. Each speck is `shapes` (offsets) in one of `colors`. With `wrap`
        the specks cross the edges, which is what makes a ground tile repeat cleanly.
        `only_on`: restrict to pixels currently of that colour."""
        rng = random.Random(seed)
        for _ in range(count):
            x, y = rng.randrange(self.w), rng.randrange(self.h)
            color = rng.choice(colors)
            shape = rng.choice(shapes) if isinstance(shapes[0][0], (tuple, list)) else shapes
            for ox, oy in shape:
                px, py = x + ox, y + oy
                if wrap:
                    px, py = px % self.w, py % self.h
                if 0 <= px < self.w and 0 <= py < self.h:
                    if only_on is None or self.img.getpixel((px, py)) == only_on:
                        self.img.putpixel((px, py), color)
        return self

    def paste(self, other: Image.Image, x: int = 0, y: int = 0) -> "Canvas":
        self.img.alpha_composite(other, (x, y))
        return self


# ---------------------------------------------------------------- transforming

def outline(img: Image.Image, color, diagonal=False) -> Image.Image:
    """One-pixel outline drawn on the transparent pixels that touch the shape. The canvas
    does not grow, so leave a one-pixel margin around the artwork."""
    src = img.load()
    out = img.copy()
    dst = out.load()
    w, h = img.size
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonal:
        steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for y in range(h):
        for x in range(w):
            if src[x, y][3] != 0:
                continue
            for dx, dy in steps:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and src[nx, ny][3] != 0:
                    dst[x, y] = color
                    break
    return out


def swap(img: Image.Image, mapping: dict) -> Image.Image:
    """Recolour: {from_colour: to_colour}. The way to make variants of one sprite."""
    out = img.copy()
    px = out.load()
    table = {_rgba(a): _rgba(b) for a, b in mapping.items()}
    for y in range(out.height):
        for x in range(out.width):
            if px[x, y] in table:
                px[x, y] = table[px[x, y]]
    return out


def mirror(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def shift(img: Image.Image, dx: int, dy: int) -> Image.Image:
    """Move the artwork inside its canvas (for a one-pixel bob between frames)."""
    out = blank(*img.size)
    out.alpha_composite(img, (0, 0))
    moved = blank(*img.size)
    moved.paste(img, (dx, dy), img)
    return moved


def scale(img: Image.Image, factor: int) -> Image.Image:
    if int(factor) != factor or factor < 1:
        raise ValueError("pixel art scales by whole numbers only")
    return img.resize((img.width * factor, img.height * factor), Image.NEAREST)


def sheet(images: list[Image.Image], cols: int, pad: int = 0) -> Image.Image:
    """Pack same-sized images into a grid, row by row."""
    w, h = images[0].size
    for i, im in enumerate(images):
        if im.size != (w, h):
            raise ValueError(f"image {i} is {im.size}, expected {(w, h)}")
    rows = (len(images) + cols - 1) // cols
    out = blank(cols * (w + pad) - pad, rows * (h + pad) - pad)
    for i, im in enumerate(images):
        out.alpha_composite(im, ((i % cols) * (w + pad), (i // cols) * (h + pad)))
    return out


# ---------------------------------------------------------------- output and checks

def save(img: Image.Image, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def preview(img: Image.Image, path: str | Path, factor: int = 8, grid: int | None = None,
            backgrounds=((58, 60, 70), (200, 205, 190))) -> Path:
    """An enlarged copy for looking at. The artwork is shown on a dark and on a light
    checkerboard (outlines and halos show differently on each), with the true-size image in
    the corner - that small one is what the player sees. `grid`: draw cell lines every
    `grid` art pixels, to judge a sheet."""
    big = scale(img, factor)
    panels = []
    for base in backgrounds:
        other = tuple(min(255, c + 14) for c in base)
        bg = Image.new("RGBA", big.size, base + (255,))
        cell = factor * 2
        for y in range(0, big.height, cell):
            for x in range(0, big.width, cell):
                if (x // cell + y // cell) % 2:
                    bg.paste(other + (255,), (x, y, min(x + cell, big.width), min(y + cell, big.height)))
        bg.alpha_composite(big)
        if grid:
            line = (255, 80, 80, 255)
            for gx in range(0, img.width + 1, grid):
                x = min(gx * factor, big.width - 1)
                bg.paste(line, (x, 0, x + 1, big.height))
            for gy in range(0, img.height + 1, grid):
                y = min(gy * factor, big.height - 1)
                bg.paste(line, (0, y, big.width, y + 1))
        # True size goes in a strip above the enlargement, so it never covers the art.
        framed = Image.new("RGBA", (big.width, big.height + img.height + 6), base + (255,))
        framed.alpha_composite(img, (3, 3))
        framed.alpha_composite(bg, (0, img.height + 6))
        panels.append(framed)
    out = Image.new("RGBA", (big.width * len(panels) + 8 * (len(panels) - 1), panels[0].height), (20, 20, 24, 255))
    for i, p in enumerate(panels):
        out.alpha_composite(p, (i * (big.width + 8), 0))
    return save(out, path)


def tile_preview(tile: Image.Image, path: str | Path, n: int = 3, factor: int = 6) -> Path:
    """The tile repeated n×n, enlarged: seams and over-distinctive features show up here."""
    rep = blank(tile.width * n, tile.height * n)
    for y in range(n):
        for x in range(n):
            rep.alpha_composite(tile, (x * tile.width, y * tile.height))
    return save(scale(rep, factor), path)


def strip_preview(frames: list[Image.Image], path: str | Path, factor: int = 8, ms: int = 180) -> Path:
    """An animated GIF of the frames, enlarged, on a neutral background."""
    out = []
    for f in frames:
        bg = Image.new("RGBA", f.size, (70, 74, 84, 255))
        bg.alpha_composite(f)
        out.append(scale(bg, factor).convert("P", palette=Image.ADAPTIVE))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out[0].save(path, save_all=True, append_images=out[1:], duration=ms, loop=0)
    return path


def check(img: Image.Image, pal: dict | None = None, max_colors: int = 16) -> list[str]:
    """Problems that should not survive: partial alpha, colours outside the palette, too
    many colours. Returns a list of plain sentences; empty means clean."""
    problems = []
    seen = {}
    px = img.load()
    for y in range(img.height):
        for x in range(img.width):
            c = px[x, y]
            if c[3] == 0:
                continue
            if c[3] != 255:
                problems.append(f"partial transparency at ({x},{y}): alpha {c[3]}")
                continue
            seen.setdefault(c, (x, y))
    if pal is not None:
        allowed = {v for k, v in pal.items() if not k.startswith("_")}
        for c, at in seen.items():
            if c not in allowed:
                problems.append("colour #%02x%02x%02x at %s is not in the palette" % (c[0], c[1], c[2], at))
    if len(seen) > max_colors:
        problems.append(f"{len(seen)} colours used; the limit for one sprite is {max_colors}")
    return problems[:20]
