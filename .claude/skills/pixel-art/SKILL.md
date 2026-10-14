---
name: pixel-art
description: Make pixel-art game assets as PNG files - sprites, characters, terrain tiles, tilesets, items, UI icons, animation frames, sprite sheets - by writing small Python scripts with the bundled pixelkit, then looking at the result and fixing it. Use this whenever the user asks for art, sprites, tiles, icons, a new look for a tile or sprite in the map designer, "像素", "美术", "素材", "贴图", "图标", or to replace placeholder shapes with real graphics, even if they never say "pixel art". Also use it to restyle, recolour, animate or re-export existing pixel assets.
---

# Pixel art

You cannot paint, but you can place pixels deliberately. Pixel art made by code goes wrong in
predictable ways - muddy colours, blurry scaling, shapes that are unreadable at real size - and
all of them are avoided by working in the order below and by **looking at every image you make**.

The kit is `pixelkit.py` at the repository root (import it; do not re-implement it). In this
repository a sprite's source is the script that draws it - `trees.py` for the trees, `seed.py`
for the starter tiles - or a `save_tile` call through the MCP server; the tileset stores the art
as text rows, and `Project.export` writes the PNG sheets to `export/`. Put previews you make
for yourself under `export/preview/`.

## The order of work

1. **Read the brief as a list of things on a grid.** Fix the cell size before anything else.
   The map designer's cell is **16 px** (`mapkit.CELL`) and is shown at whole multiples (2× when
   walking), so one art pixel is always a whole number of screen pixels. A person is 16 px tall
   and one cell is about one step: measure everything against that - a bush is about a cell, a
   tree two or three cells wide, a hut about 3×2 cells. Never mix pixel densities in one scene -
   it is the fastest way to look amateur.
2. **Choose colours from the palette** (`palette.json` at the repository root, loaded by
   `pixelkit.load_palette`). Pick a ramp per material (3-4 steps, dark to light) and one ink
   colour for outlines. If something truly needs a colour that is not there, add it to the
   palette file on purpose - do not invent hex values inside a sprite.
3. **Block the silhouette first.** Fill the shape in one flat mid-tone and look at it at 1×.
   If you cannot tell what it is from the silhouette alone, shading will not save it.
4. **Shade with the ramp, light from the top-left.** Two or three bands, not a gradient. Then
   outline. Then add the one or two details that identify the thing (berries on a bush, a
   hat band on a person).
5. **Render, preview, look.** `pixelkit.save()` writes the PNG; `pixelkit.preview()` writes an
   enlarged copy on a checkerboard. Open the preview with the Read tool and actually look.
   Go through the checklist below. Expect to fix something - the first render is a draft.
6. **Check mechanically.** `pixelkit.check()` reports colours outside the palette, partial
   transparency, and too many colours. Tiles also get `pixelkit.tile_preview()` to expose seams.
7. **Pack and wire up.** Sheets via `pixelkit.sheet()`; for map tiles, store the art in the
   tileset (`save_tile`, or the tile's entry in `seed.py`) and let `Project.export` pack it.

Show the user the preview (SendUserFile) when a batch is done; art is judged by eye, theirs
more than yours.

## Two ways to place pixels

**Text grids** for anything with character - people, items, icons. Each row is a string, each
character a palette colour, `.` is transparent. The kit checks that every row has the same
width, so a miscounted row is an error rather than a skewed sprite.

```python
from pixelkit import load_palette, from_grid, outline, save, preview
P = load_palette()
berry = from_grid([
    "........",
    "..gg....",
    ".gGGg.r.",
    "gGGGGgRr",
    "gGrGGGg.",
    ".gRrGg..",
    "..ggg...",
    "........",
], {"g": "g1", "G": "g2", "r": "r1", "R": "r2"}, P)
berry = outline(berry, P["n0"])
```

**Primitives** for organic masses and textures - `Canvas.blob()` (an ellipse shaded in bands
from a ramp), `Canvas.rect()`, `Canvas.scatter()` (seeded specks that wrap at the edges, for
tileable ground). Combine them: a tree is a trunk rect, a shaded blob, and an outline.

Use fixed seeds. The same script must produce the same image every time, or assets cannot be
regenerated after a palette change.

## What to look for in the preview

- **Readable at 1×.** The preview shows the true-size sprite in the corner. Squint at that,
  not at the enlargement.
- **No stray pixels.** A single pixel of a colour with no neighbours of that colour reads as
  noise unless it is a deliberate highlight or eye.
- **No banding.** Shade bands that hug the outline in parallel lines look like contour maps.
  Break them: let the light band be a cluster near the top-left, not a ring.
- **No pillow shading.** Light comes from one side. If the centre is bright and every edge is
  dark, it was shaded from the middle; redo it.
- **Jaggies.** Diagonals and curves should step in a regular rhythm (1,1,2,2,3 - not 1,3,1,2).
- **Outline weight.** One pixel, one ink colour, no doubled corners unless the style has them.
  Interior lines can be a dark ramp colour instead of ink ("selout") to keep shapes soft.
- **Hue shift.** Shadows drift cooler and darker, highlights warmer and lighter. A ramp that
  only changes brightness looks like plastic; the palette's ramps already shift.
- **Tiles:** in the 3×3 tile preview there must be no visible grid, and no feature so
  distinctive that its repetition draws the eye.

## Rules that are not negotiable, and why

- **Nearest-neighbour everywhere.** Any smoothing turns pixels into blur. The kit never
  resamples except by integer nearest; whatever engine shows the art must use nearest filtering too.
- **Fully opaque or fully transparent.** Half-transparent pixels are anti-aliasing by another
  name and show as halos on other backgrounds.
- **Integer scales only.** 2× and 3× are crisp; 1.5× is not pixel art any more.
- **Keep the script.** The script that draws an asset is its source. Never hand-edit a PNG
  you can regenerate.

## Characters with variants

When one sprite has to appear in many colourings or builds (say 8 body colours × 8 accent
colours × 4 builds), draw one grid per build using placeholder keys (the palette's `body`,
`body_s`, `acc`, `acc_s`) and produce the rest with `pixelkit.swap()`. Do not draw 256 sprites.

## Animation

Two to four frames is enough for an idle bob or a walk at this size. Draw frame 1, copy it, and
move only what moves - a walk is the feet and a one-pixel body bob, not a redraw. Pack frames
left to right in one row per animation; `pixelkit.strip_preview()` writes an animated GIF so
the motion can be judged.

## References

- `references/craft.md` - longer notes on ramps, dithering, readability and common repairs.
  Read when a sprite looks wrong and the checklist has not told you why.
