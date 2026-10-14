# Craft notes

Read this when a sprite looks wrong and the checklist in SKILL.md has not said why.

## Ramps

A ramp is 3-5 colours for one material, dark to light. Good ramps change hue as well as
brightness: shadows lean toward blue or purple and lose saturation, highlights lean toward
yellow and gain it. Adjacent steps should be clearly different at 1× - if two steps cannot be
told apart at true size, drop one. Fewer, bolder steps beat many subtle ones at 16 px.

Share colours between ramps. The darkest green and the darkest brown can be the same colour;
the brightest highlight on stone and on water can be the same near-white. A scene made from
24 colours that reuse each other looks unified; 60 unrelated colours look like clip art.

## Silhouette and proportion

At 16×16 a person is roughly a 6-8 px head on a 6 px body. Big heads read better than correct
anatomy. Give each kind of thing a distinct outline shape: a tree is tall with a round top, a
bush is a low wide dome, a rock is angular with a flat base. If two things share a silhouette,
colour alone has to separate them, and it will fail for some players.

## Shading

Pick the light direction once for the whole game (here: top-left) and never break it. Place
the lightest band as a small cluster where the light hits, the mid-tone over most of the form,
the dark band along the far side and underneath. Leave the bands as solid clusters. Avoid:

- **banding** - bands that follow the outline in concentric rings;
- **pillow shading** - bright centre, dark rim on every side;
- **single-pixel noise** - "texture" made of isolated pixels; use 2-3 px clusters instead.

## Outlines

A full ink outline makes sprites pop off busy ground and is the safe default for anything the
player must find quickly (characters, resources). Ground tiles get no outline. For large soft
forms, replace ink on the lit side with the darkest ramp colour ("selective outline") so the
shape does not look cut out with scissors.

## Dithering

Checkerboard dithering fakes a colour between two others. At 16 px it mostly reads as noise.
Use it only on large flat areas (water, a 32 px roof), never on characters or items.

## Tiles that repeat

Keep the base flat and scatter small clusters with wrap-around so nothing lines up on a grid
edge. Avoid any single feature big enough to notice - a lone flower repeated every 16 px
becomes wallpaper. Two or three variants of a tile, chosen by a hash of the cell position,
break repetition far better than one elaborate tile.

## Common repairs

| It looks... | Usually because | Try |
|---|---|---|
| muddy | too many similar colours | cut the ramp to 3 steps, raise contrast |
| flat | no dark band, or light band too large | add shadow under and to the lower-right |
| noisy | isolated pixels, dithering | merge into clusters, remove dither |
| blurry in game | texture filter or a non-integer scale | nearest filtering, integer scale only |
| like plastic | ramp changes brightness only | shift hue across the ramp |
| unreadable small | detail without silhouette | redo the silhouette in one flat colour first |
| cut-out | heavy ink outline on a soft form | selective outline on the lit side |
