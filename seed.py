"""The starter tileset and one demo map.

    python seed.py          # only if there is no tileset yet
    python seed.py --force  # rebuild the starter tiles (keeps other tiles)

Ground tiles are materials, listed from the lowest rank (the sea) to the highest (paving).
Each says how its surface looks and how it meets the ground below it; that meeting - what it
does just inside its border, what it spills over it - is most of the design.
Object sprites are drawn here with pixelkit.
"""
from __future__ import annotations

import random
import sys

import mapkit
import treegrow
from mapkit import MapError, Project, ROOT, image_art, pk

BANK = {"colors": ["w2", "w1", "w0"], "over": ["water"]}  # a lip of earth where turf meets water

GROUNDS = [
    # The sea. Waves come in, run up the beach, slide back and leave it wet.
    ("sea", "海", "swim", ["water", "sea"], {
        "base": "b1",
        "patches": [{"color": "b0", "scale": 70, "amount": 0.34, "fill": 0.45}],   # deeper water, as a darkening, not as holes
        "specks": [{"color": "b2", "shape": "h4", "density": 0.005, "anim": "drift"},
                   {"color": "b2", "shape": "h2", "density": 0.004, "anim": "drift"},
                   # off a river's mouth, its current carries on a little way and fades
                   {"color": "b2", "shape": "h3", "density": 0.05, "anim": "flow"},
                   {"color": "b3", "shape": "h2", "density": 0.012, "anim": "flow"}],
        "shore": {"colors": ["b2", "b2", "b2", "b2", "b2", "b2"]},
        "surf": {"foam": "f0", "water": "b2", "reach": 8, "onto": ["sand", "gravel", "mud", "platform", "shingle"]},
        # the swell: long crests parallel to the coast, coming in to break as the surf
        "water": [{"move": "surface", "color": "b2", "glint": "b3", "spacing": 22, "density": 0.045},
                  {"move": "swell", "color": "b2", "glint": "b3", "spacing": 16},
                  {"move": "ruffle", "color": "b0", "glint": "b3"}]}),
    # Ordinary water - the kind that dries. Dark and clear; ripples slide, the edge laps.
    ("water", "水", "swim", ["water"], {
        "base": "b1",
        "patches": [{"color": "b0", "scale": 60, "amount": 0.22, "fill": 0.45}],
        "specks": [{"color": "b2", "shape": "h4", "density": 0.012, "anim": "flow"},
                   {"color": "b2", "shape": "h2", "density": 0.008, "anim": "flow"},
                   {"color": "b3", "shape": "dot", "density": 0.0012, "anim": "twinkle"}],
        "shore": {"colors": ["b3", "b2", "b2", "b2"], "lap": True},
        # It runs. Toward the sea if it reaches it, and there the two waters go together: a
        # river mouth is not a shore. A pond, which reaches nothing, lies still.
        "flows": True, "joins": ["sea"],
        "face": ["f0", "b3", "f0", "b3", "b2"], "lip": "b3",   # where it falls over a step: white water
        # the wind ruffles it in the gusts, and now and then a fish rises
        "water": [{"move": "surface", "color": "b2", "glint": "b3", "spacing": 14, "density": 0.04},
                  {"move": "ruffle", "color": "b0", "glint": "b3"}, {"move": "rings", "color": "b2", "glint": "b3", "density": 0.0004}],
        "decor": [{"tile": "lily", "density": 0.03}]}),
    # A lagoon: sea water shut in behind a spit. Calm, shallow, paler where it shoals; it laps
    # and does not break.
    ("lagoon", "潟湖", "swim", ["water", "lagoon"], {
        "base": "b1",
        "patches": [{"color": "b2", "scale": 40, "amount": 0.35, "fill": 0.45}, {"color": "b0", "scale": 30, "amount": 0.15, "fill": 0.4}],
        "specks": [{"color": "b2", "shape": "h3", "density": 0.008, "anim": "drift"}, {"color": "b3", "shape": "dot", "density": 0.002, "anim": "twinkle"}],
        "shore": {"colors": ["b3", "b2", "b2"], "lap": True},
        "water": [{"move": "surface", "color": "b2", "glint": "b3", "spacing": 18, "density": 0.035},
                  {"move": "ruffle", "color": "b0", "glint": "b3"}]}),
    # A tide pool in the rock shelf: clear, still, a pale rim of rock.
    ("pool", "潮池", "swim", ["water", "pool"], {
        "base": "b2",
        "patches": [{"color": "b1", "scale": 6, "amount": 0.45, "fill": 0.5}],
        "specks": [{"color": "b3", "shape": "dot", "density": 0.01, "anim": "twinkle"}, {"color": "f0", "shape": "dot", "density": 0.002, "anim": "twinkle"}],
        "shore": {"colors": ["s3", "b3"]},
        "water": [{"move": "surface", "color": "b3", "glint": "f0", "spacing": 10, "density": 0.03},
                  {"move": "rings", "color": "b3", "glint": "f0", "density": 0.004}]}),   # spray dripping in
    # A pool in the bog: peat-black water, a glint of sky, a soft brown edge.
    ("bogpool", "沼池", "swim", ["water", "peat"], {
        "base": "b0",
        "patches": [{"color": "n0", "scale": 8, "amount": 0.4, "fill": 0.5}],
        "specks": [{"color": "b2", "shape": "h2", "density": 0.006, "anim": "twinkle"}, {"color": "b3", "shape": "dot", "density": 0.002, "anim": "twinkle"}],
        "shore": {"colors": ["w0", "w1"]},
        # gas from the peat: a bubble breaks and a ring spreads on the black water
        "water": [{"move": "surface", "color": "b1", "glint": "b2", "spacing": 16, "density": 0.025},
                  {"move": "rings", "color": "b2", "glint": "b3", "density": 0.002}]}),
    # A ford: the river runs shallow over stones and can be waded. Where a river would otherwise
    # cut a map in two, the generator puts one at the narrowest, stoniest crossing.
    ("ford", "浅滩", "walk", ["water", "ford"], {
        "base": "b2",
        "patches": [{"color": "b1", "scale": 10, "amount": 0.35, "fill": 0.5}],
        "specks": [{"color": "s3", "shape": "h2", "density": 0.03}, {"color": "s2", "shape": "dot", "density": 0.03},
                   {"color": "b3", "shape": "h3", "density": 0.02, "anim": "flow"}, {"color": "f0", "shape": "dot", "density": 0.006, "anim": "flow"}],
        "flows": True, "joins": ["water", "sea"],
        "face": ["f0", "b3", "f0", "b3", "b2"], "lip": "b3"}),
    # The rock shelf the sea has planed off at the foot of a hard coast: flat, wet, bedded,
    # bare. Walkable when the water is down.
    ("platform", "海蚀平台", "walk", ["land", "rock", "shore"], {
        "base": "s1",
        "patches": [{"color": "s0", "scale": 10, "amount": 0.4, "fill": 0.5}, {"color": "b1", "scale": 14, "amount": 0.22, "fill": 0.35}],
        "specks": [{"color": "s2", "shape": "h4", "density": 0.02}, {"color": "s0", "shape": "diag", "density": 0.006},
                   {"color": "b2", "shape": "h2", "density": 0.006, "anim": "twinkle"}],
        "edge": [{"color": "s0", "width": 2, "wobble": 1}],
        "face": ["s1", "s0", "s0"], "lip": "s2"}),
    # Mud at the water's edge, where the clay is. Dark, wet, with a glint.
    ("mud", "泥岸", "walk", ["land", "clay"], {
        "base": "w1",
        "patches": [{"color": "w0", "scale": 16, "amount": 0.35, "fill": 0.55}, {"color": "w2", "scale": 20, "amount": 0.3, "fill": 0.4}],
        "specks": [{"color": "b2", "shape": "h2", "density": 0.005, "anim": "twinkle"}, {"color": "w2", "shape": "dot", "density": 0.012}],
        "edge": [{"color": "w0", "width": 2, "wobble": 1.5}],
        "apron": [{"color": "w1", "width": 5, "wobble": 2, "density": 0.3, "shape": "h2"}],
        "face": ["w1", "w0", "w0"], "lip": "w2"}),
    # Salt marsh: the sheltered flats by a river mouth or a lagoon, where the sea comes in over
    # grass. Grey-green, pans of mud, a little sea lavender.
    ("saltmarsh", "盐沼", "walk", ["land", "marsh", "salt"], {
        "base": "g2",
        "patches": [{"color": "k0", "scale": 12, "amount": 0.35, "fill": 0.45}, {"color": "w1", "scale": 9, "amount": 0.25, "fill": 0.4},
                    {"color": "g1", "scale": 20, "amount": 0.3, "fill": 0.4}],
        "specks": [{"color": "pk1", "shape": "dot", "density": 0.012}, {"color": "g3", "shape": "v2", "density": 0.01, "anim": "gust"},
                   {"color": "b2", "shape": "h2", "density": 0.004, "anim": "twinkle"}],
        "edge": [{"color": "w1", "width": 2, "wobble": 1.5}],
        "apron": [{"color": "g2", "width": 4, "wobble": 2, "density": 0.3, "shape": "v2", "over": ["mud", "sand"]}],
        "face": ["w1", "w0", "w0"], "lip": "g3"}),
    # Wet sand. It darkens toward water.
    ("sand", "沙地", "walk", ["land"], {
        "base": "k2",
        "patches": [{"color": "k1", "scale": 22, "amount": 0.35, "fill": 0.3}],
        "specks": [{"color": "k1", "shape": "dot", "density": 0.01}, {"color": "f0", "shape": "h2", "density": 0.004}],
        "edge": [{"color": "k1", "width": 3, "wobble": 2}, {"color": "k0", "width": 2, "wobble": 1, "density": 0.45}],
        "face": ["k1", "k1", "k0"], "lip": "f0"}),
    # Shingle: the ridges storm waves throw up on a shore open to the sea. Bigger, paler,
    # rounder stones than a river leaves, lying in rows along the shore.
    ("shingle", "卵石脊", "walk", ["land", "flint", "shore"], {
        "base": "s2",
        "patches": [{"color": "s3", "scale": 8, "amount": 0.5, "fill": 0.5}, {"color": "k1", "scale": 14, "amount": 0.3, "fill": 0.4}],
        "specks": [{"color": "s4", "shape": "dot", "density": 0.03}, {"color": "s1", "shape": "dot", "density": 0.025},
                   {"color": "k2", "shape": "h2", "density": 0.02}, {"color": "s3", "shape": "h4", "density": 0.012}],
        "edge": [{"color": "s1", "width": 2, "wobble": 1.5}],
        "apron": [{"color": "s3", "width": 4, "wobble": 2, "density": 0.25, "shape": "dot", "over": ["sand"]}],
        "face": ["s2", "s1", "s0"], "lip": "s3",
        "decor": [{"tile": "pebble", "density": 0.12}]}),
    # The river flat: pebbles, flint. Busy and hard where sand is smooth and soft.
    ("gravel", "河滩", "walk", ["land", "flint"], {
        "base": "k1",
        "patches": [{"color": "s3", "scale": 12, "amount": 0.55, "fill": 0.5}, {"color": "k0", "scale": 18, "amount": 0.3, "fill": 0.35}],
        "specks": [{"color": "s2", "shape": "h2", "density": 0.022}, {"color": "s4", "shape": "dot", "density": 0.012},
                   {"color": "s1", "shape": "dot", "density": 0.014}, {"color": "k2", "shape": "h2", "density": 0.02}],
        "edge": [{"color": "s1", "width": 4, "wobble": 2, "density": 0.5, "shape": "h2"}],
        "apron": [{"color": "s3", "width": 6, "wobble": 2, "density": 0.28, "shape": "h2", "over": ["sand", "mud"]},
                  {"color": "s1", "width": 4, "wobble": 1, "density": 0.15, "over": ["sand", "mud"]}],
        "face": ["s2", "s1", "s1", "s0"], "lip": "s3",
        "decor": [{"tile": "pebble", "density": 0.18}]}),
    ("dirt", "土路", "walk", ["land", "path"], {
        "base": "w3",
        "patches": [{"color": "w4", "scale": 16, "amount": 0.3, "fill": 0.5}],
        "specks": [{"color": "w2", "shape": "h2", "density": 0.012}, {"color": "w2", "shape": "dot", "density": 0.01}],
        "edge": [{"color": "w2", "width": 2, "wobble": 1.5}], "cliff": BANK,
        "face": ["w2", "w1", "w1", "w0"], "lip": "w4",
        "decor": [{"tile": "pebble", "density": 0.04}]}),
    # Dry grassland: where the ground drains fast, faces the sun or the sea wind - pale straw
    # and olive, thin, with bare earth showing. It rustles yellow in a gust.
    ("drygrass", "荒草地", "walk", ["land", "grass", "dry"], {
        "base": "g4",
        "patches": [{"color": "w4", "scale": 22, "amount": 0.45, "fill": 0.45}, {"color": "g3", "scale": 30, "amount": 0.3, "fill": 0.35},
                    {"color": "k1", "scale": 12, "amount": 0.18, "fill": 0.35}],
        "specks": [{"color": "w3", "shape": "v2", "density": 0.014}, {"color": "k2", "shape": "dot", "density": 0.006},
                   {"color": "y2", "shape": "h3", "density": 0.1, "anim": "gust"}, {"color": "k2", "shape": "h2", "density": 0.05, "anim": "gust"}],
        "edge": [{"color": "w4", "width": 3, "wobble": 2.5}],
        "apron": [{"color": "g4", "width": 6, "wobble": 3, "density": 0.4, "shape": "v2", "over": ["sand", "gravel", "mud", "dirt"]}],
        "cliff": BANK,
        "face": ["w3", "w2", "w2", "w1"], "lip": "w4",
        "decor": [{"tile": "tuft", "density": 0.06}]}),
    # Meadow. It does not stop at a line: it thins out, blade by blade, onto whatever is below.
    ("grass", "草甸", "walk", ["land"], {
        "base": "g2",
        "patches": [{"color": "g3", "scale": 26, "amount": 0.5, "fill": 0.4}, {"color": "g1", "scale": 40, "amount": 0.2, "fill": 0.3}],
        "specks": [{"color": "g1", "shape": "v2", "density": 0.012}, {"color": "g4", "shape": "dot", "density": 0.004},
                   {"color": "g3", "shape": "v2", "density": 0.012},
                   # a gust lays the grass over and its pale side shows, in streaks along the wind
                   {"color": "g3", "shape": "h3", "density": 0.16, "anim": "gust"},
                   {"color": "g4", "shape": "h2", "density": 0.06, "anim": "gust"}],
        "edge": [{"color": "g3", "width": 3, "wobble": 2.5}, {"color": "g4", "width": 5, "wobble": 2, "density": 0.3, "shape": "v2"}],
        "apron": [{"color": "g2", "width": 7, "wobble": 3, "density": 0.5, "shape": "v2", "over": ["sand", "gravel", "mud", "dirt"]},
                  {"color": "g3", "width": 4, "wobble": 2, "density": 0.35, "shape": "v2", "over": ["sand", "gravel", "mud", "dirt"]}],
        "cliff": BANK,
        "face": ["w2", "w2", "w1", "w1", "w0"], "lip": "g3",   # a step in turf: a bank of earth under a lit edge
        "decor": [{"tile": "tuft", "density": 0.14}, {"tile": "flower", "density": 0.03}]}),
    # Blanket bog: on the flat wet tops, peat grows faster than it rots. Brown-green, red moss,
    # tufts of cotton grass that the wind combs. Slow going.
    ("bog", "泥炭沼", "walk", ["land", "peat", "marsh"], {
        "base": "w1",
        "patches": [{"color": "g1", "scale": 18, "amount": 0.45, "fill": 0.45}, {"color": "pk1", "scale": 10, "amount": 0.2, "fill": 0.4},
                    {"color": "g3", "scale": 12, "amount": 0.18, "fill": 0.35}],
        "specks": [{"color": "f0", "shape": "dot", "density": 0.012, "anim": "gust"}, {"color": "g3", "shape": "v2", "density": 0.008},
                   {"color": "w0", "shape": "h2", "density": 0.008}],
        "edge": [{"color": "w0", "width": 2, "wobble": 2, "density": 0.5}],
        "apron": [{"color": "w1", "width": 4, "wobble": 2, "density": 0.25, "shape": "dot", "over": ["grass", "heath"]}],
        "face": ["w1", "w0", "w0"], "lip": "g3"}),
    # Heath: the open hill above the woods and where the sea wind scours - low, tough, brown-green,
    # a little heather. What a steep slope in a wet climate mostly is, rather than bare rock.
    ("heath", "荒坡", "walk", ["land"], {
        "base": "k0",
        "patches": [{"color": "g2", "scale": 20, "amount": 0.5, "fill": 0.45}, {"color": "w1", "scale": 26, "amount": 0.35, "fill": 0.4}],
        "specks": [{"color": "w2", "shape": "dot", "density": 0.02}, {"color": "pk1", "shape": "dot", "density": 0.008},
                   {"color": "g3", "shape": "v2", "density": 0.006, "anim": "gust"}, {"color": "w0", "shape": "h2", "density": 0.006}],
        "edge": [{"color": "w1", "width": 3, "wobble": 2, "density": 0.4}],
        "apron": [{"color": "g1", "width": 5, "wobble": 2, "density": 0.35, "shape": "dot", "over": ["grass"]}],
        "face": ["w1", "w1", "w0", "w0"], "lip": "w2",
        "decor": [{"tile": "fern", "density": 0.05}, {"tile": "pebble", "density": 0.05}]}),
    # The floor of the old wood. Two steps darker than meadow, brown with litter, and it
    # throws its shade out onto the grass.
    ("forest", "林地", "walk", ["land"], {
        "base": "g1",
        "patches": [{"color": "g0", "scale": 34, "amount": 0.55, "fill": 0.55}, {"color": "w1", "scale": 15, "amount": 0.35, "fill": 0.3}],
        "specks": [{"color": "w1", "shape": "h2", "density": 0.012}, {"color": "w2", "shape": "dot", "density": 0.005},
                   {"color": "g0", "shape": "l", "density": 0.01}, {"color": "g2", "shape": "v2", "density": 0.004}],
        "edge": [{"color": "g0", "width": 5, "wobble": 3}],
        "apron": [{"color": "g1", "width": 8, "wobble": 3, "density": 0.55, "shape": "l", "over": ["grass"]},
                  {"color": "g0", "width": 3, "wobble": 2, "density": 0.4, "shape": "h2", "over": ["grass"]},
                  {"color": "w1", "width": 6, "wobble": 2, "density": 0.06, "over": ["grass", "sand", "gravel", "dirt"]}],
        "cliff": BANK,
        "face": ["w1", "w1", "w0", "w0", "n0"], "lip": "g2",
        "decor": [{"tile": "fern", "density": 0.12}, {"tile": "mushroom", "density": 0.03}]}),
    # Scree: broken rock lying at the angle it falls to, under faces and on the steepest slopes.
    ("scree", "碎石坡", "walk", ["land", "loose"], {
        "base": "s1",
        "patches": [{"color": "s2", "scale": 12, "amount": 0.5, "fill": 0.55}],
        "specks": [{"color": "s3", "shape": "step", "density": 0.03}, {"color": "s0", "shape": "l", "density": 0.02},
                   {"color": "s2", "shape": "h2", "density": 0.03}, {"color": "s4", "shape": "dot", "density": 0.006}],
        "edge": [{"color": "s0", "width": 2, "wobble": 2, "density": 0.5}],
        "apron": [{"color": "s2", "width": 6, "wobble": 2, "density": 0.25, "shape": "h2",
                   "over": ["grass", "heath", "forest", "sand", "gravel", "mud"]}],
        "face": ["s2", "s1", "s0"], "lip": "s3",
        "decor": [{"tile": "pebble", "density": 0.2}]}),
    # A rock ledge. A lit lip, a face on the south side, and scree fallen around its foot.
    ("mountain", "岩台", "walk", ["land", "no_build"], {
        "base": "s2",
        "patches": [{"color": "s3", "scale": 30, "amount": 0.32, "fill": 0.55}, {"color": "s1", "scale": 16, "amount": 0.2, "fill": 0.45}],
        "specks": [{"color": "s1", "shape": "h4", "density": 0.012}, {"color": "s3", "shape": "h3", "density": 0.006},
                   {"color": "s0", "shape": "diag", "density": 0.004}],     # bedding and joints, not blotches
        "rim": "s4", "face": ["s2", "s1", "s1", "s0", "s0", "n0"], "lip": "s4",
        "apron": [{"color": "s3", "width": 8, "wobble": 3, "density": 0.3, "shape": "h2", "over": ["grass", "forest", "sand", "gravel", "dirt", "mud"]},
                  {"color": "s1", "width": 6, "wobble": 2, "density": 0.2, "shape": "h2", "over": ["grass", "forest", "sand", "gravel", "dirt", "mud"]}],
        "cliff": {"colors": ["s1", "s1", "s2", "s1", "s1", "s1", "s0", "s0", "n0"], "cracks": True},
        "decor": [{"tile": "pebble", "density": 0.08}]}),
    # An outcrop of ore: hard rock stained green-blue where copper weathers out of it (verdigris,
    # malachite), rusty where there is iron with it.
    ("copper", "铜矿露头", "walk", ["land", "rock", "ore", "copper", "resource", "no_build"], {
        "base": "s2",
        "patches": [{"color": "g3", "scale": 6, "amount": 0.35, "fill": 0.5}, {"color": "b3", "scale": 8, "amount": 0.25, "fill": 0.45},
                    {"color": "o0", "scale": 10, "amount": 0.18, "fill": 0.4}],
        "specks": [{"color": "s0", "shape": "h3", "density": 0.012}, {"color": "b3", "shape": "dot", "density": 0.008, "anim": "twinkle"}],
        "rim": "s4", "face": ["s2", "g2", "s1", "s1", "s0", "n0"], "lip": "s4",
        "apron": [{"color": "g3", "width": 4, "wobble": 2, "density": 0.12, "shape": "dot", "over": ["scree", "heath", "grass"]}]}),
    # Snow that lies all year: on the high ground above the snow line, longest on the slopes that
    # face away from the sun and in the hollows. White, faintly blue in its shadows; it glints.
    ("snow", "雪地", "walk", ["land", "snow", "cold", "slow"], {
        "base": "f0",
        "patches": [{"color": "t2", "scale": 30, "amount": 0.4, "fill": 0.5}, {"color": "s4", "scale": 14, "amount": 0.2, "fill": 0.35}],
        "specks": [{"color": "b3", "shape": "dot", "density": 0.006, "anim": "twinkle"}, {"color": "t1", "shape": "h3", "density": 0.01},
                   {"color": "t2", "shape": "h2", "density": 0.05, "anim": "gust"}],
        "edge": [{"color": "t1", "width": 3, "wobble": 2.5}, {"color": "t2", "width": 5, "wobble": 3, "density": 0.4, "shape": "dot"}],
        "face": ["t1", "s3", "s2", "s1"], "lip": "f0"}),
    # A sea stack: a pillar of the cliff left standing in the sea when the rest fell.
    ("stack", "海蚀柱", "block", ["rock", "no_build"], {
        "base": "s2",
        "patches": [{"color": "s3", "scale": 10, "amount": 0.3, "fill": 0.5}],
        "specks": [{"color": "s1", "shape": "h4", "density": 0.015}],
        "rim": "s4", "face": ["s1", "s3", "s1", "s1", "s0", "n0"], "lip": "s4",
        "cliff": {"colors": ["s1", "s3", "s1", "s1", "s3", "s1", "s1", "s0", "n0"]}}),
    # The mouth of a sea cave at the foot of a cliff: dark, going in.
    ("cave", "海蚀洞", "walk", ["cave", "no_build"], {
        "base": "n0",
        "patches": [{"color": "s0", "scale": 4, "amount": 0.3, "fill": 0.45}],
        "specks": [{"color": "b1", "shape": "h2", "density": 0.006, "anim": "twinkle"}],
        "rim": "s0", "face": ["n0", "n0", "s0"], "lip": "s1"}),
    # A sea cliff: pale bands of weathering across the face.
    ("crag", "崖岸", "walk", ["land", "no_build", "seacliff"], {
        "base": "s2",
        "patches": [{"color": "s3", "scale": 30, "amount": 0.28, "fill": 0.55}, {"color": "s1", "scale": 14, "amount": 0.25, "fill": 0.5}],
        "specks": [{"color": "s1", "shape": "h4", "density": 0.012}],
        "rim": "s4", "face": ["s1", "s3", "s1", "s1", "s3", "s0"], "lip": "s4",
        "apron": [{"color": "s3", "width": 5, "wobble": 2, "density": 0.15, "shape": "h2"}],
        "cliff": {"colors": ["s1", "s3", "s1", "s1", "s3", "s1", "s3", "s1", "s1", "s1", "s3", "s1", "s1", "s1", "s0", "n0"]}}),
    # A paved floor, old; the land is taking its edges back.
    ("paving", "石铺地", "walk", ["paved"], {
        "base": "s1",
        "patches": [{"color": "s0", "scale": 18, "amount": 0.3}, {"color": "s2", "scale": 30, "amount": 0.25, "fill": 0.5}],
        "specks": [{"color": "s0", "shape": "h4", "density": 0.006}, {"color": "s2", "shape": "h2", "density": 0.004}],
        "rim": "s0", "face": ["s1", "s0", "s0"], "lip": "s2",
        "edge": [{"color": "g1", "width": 5, "wobble": 3, "density": 0.3, "shape": "v2"}]}),
]


# ---------------------------------------------------------------- sprites

def person(P, coat=("k2", "k1"), trim=("r1", "r0")):
    """Someone to measure everything else against: 16 px tall, one cell is about one step."""
    rows = [
        "...hhhh...",
        "..hhhhhh..",
        "..hssssh..",
        "..ssssss..",
        "...ssss...",
        "..cCCCCc..",
        ".cCCCCCCc.",
        ".cCtTTtCc.",
        ".cCCCCCCc.",
        ".sCCCCCCs.",
        "..cCCCCc..",
        "..cCCCCc..",
        "..ww..ww..",
        "..ww..ww..",
        "..bb..bb..",
        "..........",
    ]
    key = {"h": "w1", "s": "w4", "C": coat[0], "c": coat[1], "T": trim[0], "t": trim[1], "w": "w2", "b": "w0"}
    return pk.outline(pk.from_grid(rows, key, P), P["n0"])


def oak(P, lean=0):
    """A broad tree: the canopy is a few shaded clumps, so the highlight breaks up."""
    leaf = pk.ramp(P, "leaf")
    c = pk.Canvas(32, 42)
    c.blob(16, 21, 10, 6, leaf[:2] + leaf[1:3], bands=[0.3, 0.6, 0.9])  # the shaded underside
    c.blob(8.5 - lean, 19, 7, 6.5, leaf, bands=[0.25, 0.55, 0.85])
    c.blob(23.5 - lean, 19.5, 7, 6.5, leaf, bands=[0.32, 0.62, 0.9])
    c.blob(16, 13, 12.5, 10.5, leaf, bands=[0.2, 0.5, 0.82])
    c.blob(12 + lean, 8, 7, 5, leaf, bands=[0.15, 0.4, 0.72])
    canopy = pk.outline(c.img, P["g0"])
    t = pk.Canvas(32, 42)
    t.rect(14, 24, 4, 16, P["w1"]).rect(15, 24, 1, 16, P["w2"]).rect(17, 24, 1, 16, P["w0"])
    t.px(13, 39, P["w1"]).px(18, 39, P["w0"]).px(12, 40, P["w1"]).px(19, 40, P["w0"])
    out = pk.outline(t.img, P["w0"])
    out.alpha_composite(canopy)
    return out


def pine(P):
    """Three tiers, each lit on the left and shaded under its own skirt."""
    c = pk.Canvas(24, 42)
    c.rect(11, 32, 3, 8, P["w1"]).rect(11, 32, 1, 8, P["w2"])
    for top, bottom, half in [(20, 33, 10), (11, 24, 8), (2, 15, 6)]:
        for y in range(top, bottom + 1):
            h = max(1, round(half * (y - top + 1) / (bottom - top + 1)))
            for x in range(12 - h, 12 + h):
                t = (x - (12 - h)) / (2 * h)
                name = "g0" if y >= bottom - 1 or t > 0.72 else "g3" if t < 0.22 else "g2" if t < 0.48 else "g1"
                c.px(x, y, P[name])
    return pk.outline(c.img, P["g0"])


def grasses(P) -> list:
    """The grass that grows on the meadows, short and long, green and dry. Not sprites: each
    says how it grows, and it is grown over its cells (render._blades) - a cover packed with
    blades, its edge wandering past the cell, long grass standing over the cell above."""
    green, dry = ["g1", "g2", "g3", "g3", "g4"], ["w2", "w3", "w4", "g4", "k2"]

    def tile(tid, name, colours, under, height, density, spill, tags):
        return {"id": tid, "name": name, "layer": "plant", "pass": "walk", "tags": ["plant", "cuttable"] + tags,
                "grow": {"height": height, "density": density, "colors": colours, "under": under, "spill": spill}}
    bloom = {"bloom": ["y2", "pk2", "f0", "o1", "y2"], "blooming": 0.3}
    return [tile("shortgrass", "短草", green[:3], "g2", [2, 4], 0.28, 1, ["short"]),
            {**tile("flowergrass", "花草", green[:3], "g2", [2, 5], 0.28, 1, ["short", "flowers"]),
             "grow": {"height": [2, 5], "density": 0.28, "colors": green[:3], "under": "g2", "spill": 1, **bloom}},
            {**tile("flowergrass_long", "长花草", green, "g1", [6, 12], 0.22, 4, ["long", "hide", "slow", "flowers"]),
             "grow": {"height": [6, 12], "density": 0.22, "colors": green, "under": "g1", "spill": 4,
                      "bloom": bloom["bloom"], "blooming": 0.16}},
            tile("longgrass", "长草", green, "g1", [6, 12], 0.22, 4, ["long", "hide", "slow"]),
            tile("shortgrass_dry", "短枯草", dry[:3] + ["g4"], "w4", [2, 4], 0.24, 1, ["short", "dry"]),
            tile("longgrass_dry", "长枯草", dry, "w3", [6, 13], 0.2, 4, ["long", "hide", "slow", "dry", "burns"])]


def covers() -> list:
    """What grows on the other grounds - each where it would really grow (geo._cover says
    where). Grown like grass, in one of four forms."""
    def plant(tid, name, tags, **grow):
        return {"id": tid, "name": name, "layer": "plant", "pass": "walk", "tags": ["plant"] + tags, "grow": grow}
    return [
        # the wood floor: ferns in the damp shade, moss where it is wetter still
        plant("fernbed", "蕨丛", ["cuttable", "hide"], form="fronds", height=[6, 11], density=0.035,
              colors=["g0", "g1", "g2", "g3", "g4"], under="g1", spill=3),
        plant("moss", "苔藓", ["soft"], form="crust", height=[1, 1], width=2, density=0.3,
              colors=["g2", "g3"], under="g1", spill=1),
        # the heath: heather, bracken on the deeper soil, gorse where it is dry and sunny
        plant("heather", "石南", ["flowers"], form="tufts", height=[2, 4], density=0.22,
              colors=["w1", "g1", "g1"], under="w1", spill=2, bloom=["pk1", "pk2", "pk1"], blooming=0.55),
        plant("bracken", "欧洲蕨", ["cuttable", "hide", "slow", "burns"], form="fronds", height=[7, 12], density=0.04,
              colors=["w1", "g1", "g2", "w2", "g3"], under="w1", spill=4, bloom=["o0", "w2"], blooming=0.25),
        plant("gorse", "荆豆", ["thorny", "slow", "burns", "flowers"], form="tufts", height=[3, 6], density=0.18,
              colors=["g0", "g0", "g1"], under="w1", spill=2, bloom=["y1", "y2"], blooming=0.45),
        # dunes behind the beach: marram, which holds the sand
        plant("marram", "滨草", ["binds_sand"], form="blades", height=[5, 10], density=0.12,
              colors=["k1", "g3", "g4", "k2"], under="k2", spill=3),
        # by still and slow water: reeds, and rushes a little further off
        plant("reeds", "芦苇", ["cuttable", "hide", "slow", "thatch"], form="blades", stiff=True, height=[10, 15], density=0.16,
              colors=["g1", "g2", "g3", "g4"], under="w1", spill=3, bloom=["w1", "w0"], blooming=0.5),
        plant("rushes", "灯心草", ["cuttable"], form="blades", height=[4, 7], density=0.26,
              colors=["g0", "g1", "g2"], under="w1", spill=2),
        # the salt marsh: samphire by the creeks, sea lavender on the higher marsh
        plant("samphire", "盐角草", ["edible", "salt"], form="tufts", height=[1, 3], width=2, density=0.28,
              colors=["g1", "g2", "pk1"], under="w1", spill=1),
        plant("sealavender", "海薰衣草", ["flowers", "salt"], form="blades", height=[3, 5], density=0.2,
              colors=["g1", "g2"], under="g2", spill=1, bloom=["pk2", "f0"], blooming=0.5),
        # the bog: cotton grass and red-and-green sphagnum
        plant("cottongrass", "羊胡子草", ["peat"], form="blades", height=[4, 7], density=0.16,
              colors=["g1", "g2", "g3"], under="w1", spill=2, bloom=["f0", "f0", "k2"], blooming=0.5),
        plant("sphagnum", "泥炭藓", ["peat", "soft"], form="crust", height=[1, 1], width=2, density=0.3,
              colors=["pk1", "g3", "g2", "pk2"], under="w1", spill=1),
        # above the trees and below the snow: little cushions that hug the rock
        plant("alpine", "高山垫状植物", ["cold"], form="tufts", height=[1, 2], width=2, density=0.12,
              colors=["g1", "g2"], under="s2", spill=1, bloom=["f0", "pk2"], blooming=0.4),
        # rock: lichen; and thrift on the shingle out of the waves' reach
        plant("lichen", "地衣", ["rock"], form="crust", height=[1, 1], width=2, density=0.07,
              colors=["o1", "g4"], under="s2", spill=0),
        plant("thrift", "海石竹", ["flowers", "salt"], form="tufts", height=[1, 3], width=2, density=0.12,
              colors=["g1", "g2"], under="s2", spill=1, bloom=["pk2", "pk1"], blooming=0.55),
    ]


def bush(P, berries=True):
    c = pk.Canvas(20, 15)
    c.blob(10, 8.5, 8.5, 5.5, pk.ramp(P, "leaf"), bands=[0.22, 0.5, 0.82])
    c.blob(6, 6.5, 4, 3, pk.ramp(P, "leaf"), bands=[0.1, 0.35, 0.7])
    if berries:
        for x, y in [(5, 9), (10, 6), (13, 10), (8, 11), (15, 7)]:
            c.px(x, y, P["r2"]).px(x + 1, y, P["r1"]).px(x, y + 1, P["r1"])
    return pk.outline(c.img, P["g0"])


def boulder(P):
    c = pk.Canvas(22, 17)
    c.blob(11, 9.5, 9.5, 7.5, pk.ramp(P, "stone") + [P["s4"]], bands=[0.18, 0.42, 0.68, 0.9])
    c.blob(15, 11, 4.5, 4, pk.ramp(P, "stone")[:3], bands=[0.4, 0.75])
    for x in range(22):  # sits flat on the ground
        for y in (15, 16):
            c.img.putpixel((x, y), (0, 0, 0, 0))
    return pk.outline(c.img, P["n0"])


def hut(P):
    """Sized against a 16 px person: the door is a little taller than they are, and the
    house covers three cells of ground."""
    c = pk.Canvas(56, 56)
    c.rect(7, 30, 42, 24, P["w4"])
    for x in range(7, 49, 5):
        c.rect(x, 30, 1, 24, P["w3"])
    c.rect(7, 30, 42, 3, P["w2"]).rect(7, 52, 42, 2, P["w2"])           # the eave's shadow; the sill
    c.rect(22, 35, 12, 19, P["w0"]).rect(23, 36, 10, 18, P["w1"])        # door frame and door
    c.rect(27, 36, 1, 18, P["w0"]).px(31, 45, P["y1"]).px(31, 46, P["y1"])
    for wx in (10, 38):                                                   # windows
        c.rect(wx, 36, 8, 8, P["w0"]).rect(wx + 1, 37, 6, 6, P["b3"]).rect(wx + 1, 37, 6, 2, P["b2"])
        c.rect(wx + 4, 37, 1, 6, P["w0"]).rect(wx + 1, 40, 6, 1, P["w0"])
    c.rect(38, 5, 5, 9, P["s2"]).rect(38, 5, 2, 9, P["s3"]).rect(37, 4, 7, 2, P["s1"])   # chimney
    for y in range(4, 32):                                                # thatch, lit from the left
        half = min(27, 4 + (y - 4))
        for x in range(28 - half, 28 + half):
            t = (x - (28 - half)) / (2 * half)
            row = (y + (x // 6) % 2) % 4
            name = "w2" if row == 0 else "w4" if t < 0.28 and row == 2 else "y1" if t < 0.5 and row == 1 else "w3"
            c.px(x, y, P[name])
    return pk.outline(c.img, P["n0"])


def grid(P, rows, key, outline=None):
    img = pk.from_grid(rows, key, P)
    return pk.outline(img, P[outline]) if outline else img


def small_sprites(P) -> dict:
    """Decor: a few pixels each, no outline, so they sit in the ground instead of on it."""
    def flower(petal, heart):
        return grid(P, [".p.", "pyp", ".p.", ".s.", ".s."], {"p": petal, "y": heart, "s": "g1"})
    return {
        "tuft": [grid(P, ["l...l.", ".l.l..", ".dld.l", "..d.d."], {"l": "g4", "d": "g1"}),
                 grid(P, [".l..", "l.l.", ".dl.", ".d.."], {"l": "g3", "d": "g1"})],
        "flower": [flower("f0", "y1"), flower("y2", "o1"), flower("f0", "g4")],  
        "fern": [grid(P, ["l...l", ".l.l.", "d.l.d", ".dld.", "..d.."], {"l": "g2", "d": "g0"}),
                 grid(P, [".l.l.", "..l..", "d.l.d", ".ddd."], {"l": "g3", "d": "g0"})],
        "mushroom": [grid(P, [".rr.", "rfrr", ".kk.", ".kk."], {"r": "r1", "f": "f0", "k": "k2"}),
                     grid(P, [".oo.", "ooyo", ".k.."], {"o": "o1", "y": "y2", "k": "k2"})],
        "pebble": [grid(P, [".hh.", "hmmd", ".dd."], {"h": "s4", "m": "s3", "d": "s1"}),
                   grid(P, ["hm", "md"], {"h": "s3", "m": "s2", "d": "s1"})],
        "lily": [grid(P, [".lll.", "llLll", ".ll.."], {"l": "g2", "L": "g3"}),
                 grid(P, [".ll.", "lpLl", ".ll."], {"l": "g2", "L": "g3", "p": "f0"})],
    }


SPAWN = (["....prrrr.", "....prrrrr", "....prrrr.", "....prr...", "....p.....", "....p.....", "....p.....",
          "....p.....", "...sps....", "..sssss..."], {"p": "w3", "r": "r2", "s": "s2"})


def recipe(P, spec):
    return mapkit.recipe_art(spec, P)


def seed_tiles(project: Project) -> int:
    P = project.palette()
    art = lambda img: image_art(img, P)  # noqa: E731
    grown = lambda kind, s, size=1.0: treegrow.tree(P, kind, s, size=size)  # noqa: E731  (trees grown from a skeleton)
    ages = (1.0, 0.84, 0.7, 0.92)   # one kind at different ages, so a wood is not all one height
    small = small_sprites(P)
    decor_names = {"tuft": "草丛", "flower": "野花", "fern": "蕨", "mushroom": "蘑菇", "pebble": "石子", "lily": "睡莲"}
    swaying = {"tuft", "flower", "fern"}  # they stand still until a gust comes through
    # Decor first: the materials refer to it.
    objects = [{"id": k, "name": decor_names[k], "pass": "walk", "tags": ["decor"], "flip": True,
                "motion": "gust" if k in swaying else "none", "variants": [art(i) for i in v]} for k, v in small.items()]
    objects += [
        {"id": "tree", "name": "树", "pass": "block", "tags": ["resource", "wood"], "shadow": True, "flip": False,
         "jitter": 3, "motion": "sway", "variants": [art(grown("oak", 0)), art(grown("oak", 1)), art(grown("pine", 0))]},
        {"id": "berry_bush", "name": "浆果丛", "pass": "walk", "tags": ["resource", "berry"], "shadow": True,
         "flip": False, "jitter": 2, "motion": "sway", "variants": [art(bush(P))]},
        {"id": "rock", "name": "岩石", "pass": "walk", "tags": ["resource", "stone"], "shadow": True, "flip": False,
         "jitter": 2, "variants": [art(boulder(P))]},
        {"id": "hut", "name": "小屋", "pass": "block", "tags": ["structure"], "shadow": True, "footprint": [3, 2],
         "variants": [art(hut(P))]},
        # Someone to measure everything else against: one cell is about one step.
        {"id": "person", "name": "人", "pass": "walk", "tags": ["marker", "npc"], "shadow": True, "flip": True, "motion": "bob",
         "variants": [art(person(P)), art(person(P, ("k1", "k0"), ("b2", "b1")))]},
        {"id": "spawn", "name": "出生点", "pass": "walk", "tags": ["marker"],
         "variants": [art(grid(P, *SPAWN, outline="n0"))]},
        # trees by kind, for the generator to put where each grows (geo._trees)
        {"id": "oak", "name": "橡树", "pass": "block", "tags": ["resource", "wood", "hardwood", "tree"], "shadow": True, "flip": False,
         "jitter": 3, "motion": "sway", "variants": [art(grown("oak", s, ages[s])) for s in range(4)]},
        {"id": "pine", "name": "松", "pass": "block", "tags": ["resource", "wood", "resin", "tree"], "shadow": True, "flip": False,
         "jitter": 3, "motion": "sway", "variants": [art(grown("pine", s, ages[s])) for s in range(3)] + [art(grown("pine_old", s, ages[s * 3])) for s in range(2)]},
        {"id": "birch", "name": "白桦", "pass": "block", "tags": ["resource", "wood", "bark", "tree"], "shadow": True, "flip": False,
         "jitter": 3, "motion": "sway", "variants": [art(grown("birch", s, ages[s])) for s in range(3)]},
        {"id": "alder", "name": "桤木", "pass": "block", "tags": ["resource", "wood", "tree", "wet"], "shadow": True, "flip": False,
         "jitter": 3, "motion": "sway", "variants": [art(grown("alder", s, ages[s])) for s in range(3)]},
        {"id": "windbent", "name": "风压树", "pass": "block", "tags": ["resource", "wood", "tree"], "shadow": True,
         "jitter": 2, "motion": "sway", "variants": [art(grown("windbent", s, ages[s])) for s in range(3)]},
        {"id": "willow", "name": "柳", "pass": "block", "tags": ["resource", "wood", "withy", "tree", "wet"], "shadow": True,
         "flip": False, "jitter": 3, "motion": "sway", "variants": [art(grown("willow", s, a)) for s, a in ((0, 1.0), (1, 0.84), (3, 0.72))]},
    ]
    for t in objects:
        project.save_tile({**t, "layer": "object"})
    for t in grasses(P) + covers():
        project.save_tile(t)
    # Twice over: a ground may name one that comes later (the sea's waves run onto the sand).
    for later in (False, True):
        for rank, (tid, name, pas, tags, look) in enumerate(GROUNDS):
            look = look if later else {k: v for k, v in look.items() if k not in ("surf", "apron")}
            project.save_tile({"id": tid, "name": name, "layer": "ground", "pass": pas, "tags": tags, "rank": rank, "look": look})
    return len(objects) + len(GROUNDS) + len(grasses(P)) + len(covers())


# ---------------------------------------------------------------- the demo map

def _noise(w, h, step, rng):
    """Smooth value noise in 0..1: random lattice, bilinear in between."""
    gw, gh = w // step + 2, h // step + 2
    lat = [[rng.random() for _ in range(gw)] for _ in range(gh)]
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            fx, fy = x / step, y / step
            ix, iy, tx, ty = int(fx), int(fy), fx % 1, fy % 1
            top = lat[iy][ix] * (1 - tx) + lat[iy][ix + 1] * tx
            bot = lat[iy + 1][ix] * (1 - tx) + lat[iy + 1][ix + 1] * tx
            row.append(top * (1 - ty) + bot * ty)
        out.append(row)
    return out


def seed_demo(project: Project, name="demo", width=48, height=32, seed=7) -> None:
    """A piece of the valley at walking scale, about a step to a cell: a clearing, a stream you
    could wade, a pond, a rock ledge. Not a continent squeezed into a screen."""
    wob = _noise(width, height, 4, random.Random(seed))
    m = project.create_map(name, width, height, fill="grass")

    def patch(m, tile, cx, cy, rx, ry):
        """An uneven oval of ground."""
        for y in range(height):
            for x in range(width):
                if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1 + 0.7 * (wob[y][x] - 0.5):
                    m.layers["ground"][y][x] = tile

    def trail(m, tiles, tile, points, wide):
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            mapkit.draw_line(m, tiles, tile, x0, y0, x1, y1, wide)

    def build(m, tiles):
        for cx, cy, rx, ry in [(5, 9, 8, 11), (22, 1, 16, 4.5), (41, 27, 9, 6.5), (4, 29, 7, 4), (44, 3, 6, 3)]:
            patch(m, "forest", cx, cy, rx, ry)
        stream = [(31, 0), (32, 5), (28, 10), (27, 15), (31, 20), (35, 25), (34, 31)]
        trail(m, tiles, "sand", stream, 4)
        trail(m, tiles, "water", stream, 2)
        patch(m, "sand", 18, 24, 5.5, 3.8)
        patch(m, "water", 18, 24, 4, 2.6)
        patch(m, "mountain", 41, 11, 5, 3.2)
        mapkit.fill_rect(m, tiles, "sand", 25, 15, 5, 2)             # a shallow place to cross
        # The valley as it is before anyone: no paths, nothing built. And the map only, for
        # now: nothing that is interacted with on its own (trees, rocks, people, memory stones)
        # is placed. What the ground sprinkles on itself stays.

    project.edit(m.name, build)


def seed_coast(project: Project, name="coast", width=48, height=32, seed=11) -> None:
    """The shore: meadow thinning onto a beach, the sea working at the sand, a stream coming
    down to it over mud and stones, a ledge of rock at the water."""
    wob = _noise(width, height, 5, random.Random(seed))
    m = project.create_map(name, width, height, fill="grass")

    def build(m, tiles):
        g = m.layers["ground"]
        # The stream first, all the way down; then the shore is laid over it, so the banks of
        # mud and stones stop at the beach and only the water itself goes on into the sea.
        stream = [(30, 0), (29, 7), (32, 13), (31, 19), (33, 31)]
        for tile, wide in (("gravel", 6), ("mud", 4), ("water", 2)):
            for (x0, y0), (x1, y1) in zip(stream, stream[1:]):
                mapkit.draw_line(m, tiles, tile, x0, y0, x1, y1, wide)
        for y in range(height):
            for x in range(width):
                edge = 17 + 3.2 * (wob[y][x] - 0.5) * 2 + 1.5 * __import__("math").sin(x / 5.0)
                if y > edge + 5.5:
                    g[y][x] = "sea"
                elif y > edge and g[y][x] == "grass":
                    g[y][x] = "sand"
        for cx, cy, rx, ry in [(7, 4, 9, 6), (40, 3, 10, 5), (24, 0, 7, 3)]:
            for y in range(height):
                for x in range(width):
                    inside = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1 + 0.7 * (wob[y][x] - 0.5)
                    if inside and g[y][x] == "grass":  # the wood does not grow over the stream
                        g[y][x] = "forest"
        for y in range(height):
            for x in range(width):
                if ((x - 10) / 5.5) ** 2 + ((y - 22) / 2.6) ** 2 <= 1 + 0.5 * (wob[y][x] - 0.5):
                    g[y][x] = "mountain"

    project.edit(m.name, build)


def main():
    mapkit.utf8_stdio()
    project = Project()
    force = "--force" in sys.argv
    if project.tileset()["tiles"] and not force:
        print("图块集已存在，未改动（--force 可重建起始图块）")
        return
    print("写入起始图块", seed_tiles(project), "个 ->", project.tileset_path)
    for name, make in (("demo", seed_demo), ("coast", seed_coast)):
        try:
            make(project)
            print("写入示例地图", name)
        except MapError as e:
            print("示例地图：", e)


if __name__ == "__main__":
    main()
