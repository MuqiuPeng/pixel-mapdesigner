---
name: map-geography
description: Make, tune and review maps so the land reads as real - terrain grown by uplift and erosion, rivers, coasts, snow, biomes, vegetation and trees placed by the laws that put them there - using this repository's generator (geo.py), its design files (designs/*.json) and the map designer's MCP tools. Use this whenever the user wants a map generated, a terrain or landscape made or changed, an island, valley, coast, river, mountain, forest, biome or "地形", "地貌", "地图生成", "随机地图", "岛", "河流", "海岸", "生物群系", "植被", "树林", "雪线", or says a map looks wrong, fake, too regular or too blocky, even if they never say "geography". Also use it before editing geo.py or render.py's terrain, water or plant code.
---

# Map geography

A generated map convinces when every feature on it is where something made it be. Do not draw
shapes and do not tune numbers until it "looks fine": find the process that makes the feature
and model that process, in the order nature runs it. Real land is the reference, and the user's
eye is the judge: **look at every map you make**, at both scales, before saying it is done.

## The order things happen in

Each layer is made from the ones before it; never let a later one decide an earlier one.

1. **Land.** Uplift against erosion (`geo._grow`: stream power solved down the drainage,
   hillslope creep, valley floors filled). A designed map lays out the uplift, the hard rock
   and the coast in `designs/<name>.json`; a random one takes them from noise. Real valleys
   (`dem/`, optional) are calibration targets, never copied.
2. **Water.** Hollows filled, flow down the steepest way, rivers where enough gathers.
3. **What water does.** Channels drawn by hydraulic law (`geo._rivers`), bars and levees,
   backswamps, the river surface stepping only at knickpoints.
4. **The coast**, by rock and exposure (`geo._coast_and_bog`).
5. **High ground**: snow line, ore, bog (`geo._high_ground`).
6. **Climate fields and biomes** (`geo._biomes`): warmth, wetness, distance to the sea, slope -
   smooth fields - turned into biome regions by a table; each biome is a recipe.
7. **What grows**: plant cover per biome and per ground (`_cover`), trees, shrubs, boulders (`_trees`).
8. **Walkability**: cliffs only where the ground is near vertical; then `connect` opens ramps,
   lays fords and moves a blocking tree until the whole map can be walked.

`references/laws.md` has the law behind every feature, the numbers the generator is
calibrated to, and the failures seen so far with their fixes. Read it before changing any rule.

## Making a map

- **Random**: `geo.make_map(project, name, width, height, seed, climate, at)` - `at` is one of
  `geo.PLACES` (mouth, river, gorge, scarp, bay, headland, upland) or valley coordinates.
  Through MCP: `generate_map`. Width and height are cells (16 px each, one cell about a step).
- **Designed** (a map someone will play and wants tuned): write or edit `designs/<name>.json`
  and generate with `design=<name>`, `at="whole"`. Tuning is editing that file and growing
  again - move a peak, widen a bench, add a spit - never painting over the result. Keep the
  design's names plain geography; story meaning belongs to the game, not to the terrain file.
- **Climates**: oceanic, dry, cold - they move the snow line, the share of dry grass and of
  bog, how many streams there are and where biomes fall.

## Checking a map

1. Read the report `make_map` returns: it says what is where and why.
2. Look at the whole map (`render_map`, or the editor's whole view) - regions, river network,
   coast - and at walking scale (`render_map` with a region, or the editor's walk view) -
   borders, plants, trees, falls.
3. Run `validate_map`, and the tests: `python -m unittest test_geo test_mapkit`.
4. Compare with the checklist in `references/laws.md`. If something looks invented, a
   process is missing or wrong - fix the process, not the cells.

When you change a rule in geo.py or render.py, regenerate, look again, and keep the tests
green; add a test for the law you added (what must hold, not what the picture looks like).
