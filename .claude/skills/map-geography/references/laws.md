# The laws, the numbers, the failures

## Land

| Feature | Law | In the code |
|---|---|---|
| Relief | Uplift balanced by river incision: dh/dt = U - K A^m S (m = 0.5, n = 1), solved implicitly from the sea up the drainage (Braun & Willett 2013); hillslope creep as diffusion | `geo._grow` |
| Valley floors | What a river brings down settles where it slows: near a big river the land lies nearly level with it, easing out up the sides | end of `_grow` |
| Hard rock | Wears slower: stands as ledges, benches, headlands; rivers steepen crossing it | `hard` field, `Kratio` |
| Drowned mouths | Sea level rose after the valleys were cut: the sea lies in the lower valleys | `_drown` |
| Steepness | Measured as local gradient in real degrees (map `scale`: metres per cell and per relief unit). A cliff is only where the ground is steeper than 40 degrees | `mapkit.gradient`, `CLIFF_DEG` |
| Scree | Falls from rock and rolls downhill only, an uneven distance - never an even ring round the rock | `place`, the `fell` loop |

Calibration (generated valley vs. measured Glen Brittle): steep > 30 degrees about 12.6% (real
12.7%), flat < 5 degrees about 8% (real 13% - real glens are glacial U-troughs, the model has
no ice), hypsometric integral 0.33 (real 0.33). Keep relief in proportion to width: about one
tenth of the valley's width; a small island with tall relief turns into cliffs everywhere.

## Water

| Feature | Law | In the code |
|---|---|---|
| Course | Reaches of the channel network, grid corners cut into curves; tributaries join at a slant | `_rivers`, `_chaikin` |
| Width | Hydraulic geometry: width ~ discharge^0.5, never narrower downstream, steps up at a confluence | `_rivers` |
| Meanders | Only on a floodplain and where the river is slow; wavelength about 11 widths, skewed; straight in steep, narrow valleys; none at reach ends so confluences meet | `_rivers` |
| Surface | Falls downstream only; level in pools, dropping at knickpoints (hard rock, high stream power) and only where enough water comes from upstream - not at a spring | `_rivers` |
| Waterfalls | Drawn only where flowing water is above the step AND below it AND the flow crosses the step. Water above with no water below is a rock wall; a step the river runs along has no face | `render._ground`, `_falls` |
| Bars | Point bar on the inside of a bend, sized by curvature; the outside is the cut bank; alternate bars on straights. Grain by stream power (slope x discharge^0.5): gravel strong, sand, mud slow; finer away from the channel | `_rivers`, `place` |
| Levee, backswamp | Right by the river the bank is a little higher and dry (levee, wooded); mud lies behind it, in low hollows, in patches - never as an even band | `place` |

## Coast

| Feature | Law |
|---|---|
| Wave-cut platform | In front of hard rock meeting the sea; flat, at the water |
| Tide pools | In the platform, away from its outer edge |
| Sea stacks | Just off the platform, where the cliff fell back; well above the water |
| Sea caves | At the foot of the tallest faces |
| Shingle ridges | On the most exposed soft shores, in rows along the shore; sheltered bays keep their sand |
| Spit and lagoon | A spit grows downdrift from a headland (with a westerly wind, eastward); the water it shuts off is a lagoon, with an inlet at the free end |
| Salt marsh | Low, flat, sheltered ground by a river mouth or lagoon, drained by branching creeks |

## High ground and climate

| Feature | Law |
|---|---|
| Snow | Above the snow line (oceanic 86% of the valley's height, dry 93%, cold 62%); lower on slopes facing away from the sun; not on faces steeper than about 40 degrees. The named highest peak must be the highest |
| Bog | Flat, wet tops in a wet climate; its pools lie along the contours (long across the slope, narrow down it) |
| Alpine cushions | Just below the snow line |
| Ore | Where the design puts a deposit and hard rock is bare enough to show it |

## Biomes (after Minecraft)

Smooth fields first - warmth (sea-level warmth for the climate, less with height, more on the
sunny side), wetness (rank of damp), distance to the sea, slope - then a table, then a
majority pass so biomes come in whole regions (most cells share all four neighbours' biome).

- cold: alpine meadow (dry) / taiga (wet); cool: moor (dry) / taiga (wet);
  mild: dry plains < plains (flower meadow on sunny patches) < forest (birch forest on the
  windward edge) < dark forest (wettest and sheltered)
- the sea wind keeps the coast open (moor); steep open slopes are moor; by the river,
  riverside; on mud away from it, swamp
- from the ground: beach, stony shore, salt marsh, peat bog, stony peaks, snowy slopes

Each biome is a recipe in `geo.BIOMES`: ground, plants in patches, trees by kind and density.

## What grows

- **Plants are a cover, not objects**: a layer per cell, drawn as a continuous growth whose edge
  wanders past the cell; long grass stands up over the cell above. Forms: blades (grass,
  reeds - stiff), fronds (fern, bracken), tufts (heather, gorse, thrift), crust (lichen, moss -
  still, each patch its own colour). Flowers grow in the grass, at blade tips, thinning out at a
  patch's edge.
- **Trees by place**: alder by water; pine high up, on hard rock and thin soil; birch at a wood's
  edge; wind-bent trees at the tree line and on the sea wind; oak in sheltered lowland. A tree's
  crown spans 2-3 cells, so about one tree per 5-10 forest cells already closes the canopy -
  leave clearings. Never shut a cell in.
- **One light**: sprites lit from the top left are never mirrored; every shadow falls the same
  way (`render.SUN`); time of day is that one value.

## Failures seen, and what fixed them

| It looked... | Because | Fix |
|---|---|---|
| Like a copied real map | Terrain taken from a DEM | Grow it; use DEMs only to calibrate |
| Cliffs everywhere | Cliff decided by tier step, not gradient; or relief too tall for the width | Gradient in real degrees; relief ~ width / 10 |
| Every rock walled in | A material drew its own rim and face against every neighbour | On height maps, material rims and faces only against still water |
| Benches as perfect ellipses ringed with cliff | Bench cut with a hard edge | Ragged edge, gentle easing except where a cliff is wanted; connect the far side to a ridge |
| River lakes and threads, grid zigzags | Width from noise, path along D8 directions | Hydraulic width, smoothed course |
| Falls in gentle reaches, at springs | Every tier crossing drawn as a fall | Pools and knickpoints; falls need water upstream |
| Waterfall with no water above | Face colour taken from whatever was above | Fall only with water above and below and flow across |
| Even mud band along rivers | "Wet and near the river" rule | Levee by the river, backswamp behind it in patches |
| Even scree ring | Scree within N cells of rock | Downhill only, uneven reach |
| Speckled biomes | Picking per cell | Smooth fields, then a majority pass |
| Whole map under trees | Too many trees for crowns 2-3 cells wide | ~1 tree per 5-10 cells, clearings by noise |
| Trees lit from different sides | Random mirroring of shaded sprites | No mirroring; variants instead |
| Too few kinds of land | One material per climate band | Each ground gets its own covers; biomes mix them |
