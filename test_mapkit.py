"""python -m unittest discover -s tools/mapdesigner"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import mapkit
import render
import seed
from mapkit import Map, MapError, Project

SAND = {"base": "k2", "specks": [{"color": "k1", "shape": "dot", "density": 0.01}]}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = Project(data_dir=Path(self.tmp.name) / "maps", export_dir=Path(self.tmp.name) / "out")
        seed.seed_tiles(self.p)
        self.tiles = self.p.tiles()
        self.pal = self.p.palette()

    def tearDown(self):
        self.tmp.cleanup()


class Art(Base):
    def test_starter_sprites_are_clean_and_round_trip(self):
        grounds = [t for t in self.tiles.values() if t["layer"] == "ground"]
        self.assertEqual([t["id"] for t in sorted(grounds, key=lambda t: t["rank"])],
                         ["sea", "water", "lagoon", "pool", "bogpool", "ford", "platform", "mud", "saltmarsh", "sand",
                          "shingle", "gravel", "dirt", "drygrass", "grass", "bog", "heath", "forest", "scree", "mountain", "copper", "snow", "stack",
                          "cave", "crag", "paving"])
        self.assertEqual(len(self.tiles["tree"]["variants"]), 3)
        for t in self.tiles.values():
            for art in t.get("variants", []):
                img = mapkit.art_image(art, self.pal)
                self.assertEqual(mapkit.pk.check(img, self.pal, max_colors=24), [], t["id"])
                self.assertEqual(mapkit.art_image(mapkit.image_art(img, self.pal), self.pal).tobytes(), img.tobytes())

    def test_recipe_is_deterministic_sized_and_in_palette(self):
        recipe = {"size": [24, 30], "fill": "y1", "ops": [
            {"op": "scatter", "seed": 3, "count": 8, "colors": ["y2"], "shapes": ["h2", "dot"]},
            {"op": "blob", "cx": 12, "cy": 12, "rx": 6, "ry": 5, "ramp": "stone"},
            {"op": "rect", "x": 0, "y": 0, "w": 2, "h": 2, "color": "w1"}]}
        a, b = mapkit.recipe_art(recipe, self.pal), mapkit.recipe_art(recipe, self.pal)
        self.assertEqual(a, b)
        self.assertEqual((len(a["rows"][0]), len(a["rows"])), (24, 30))
        self.assertTrue(set(a["key"].values()) <= {"y1", "y2", "w1", "s0", "s1", "s2", "s3"})
        self.assertEqual(len(mapkit.recipe_art({"fill": "g2"}, self.pal)["rows"]), 16)

    def test_bad_art_is_refused(self):
        with self.assertRaises(MapError):
            mapkit.art_image({"rows": ["aaa", "aa"], "key": {"a": "g2"}}, self.pal)
        with self.assertRaises(MapError):
            mapkit.art_image({"rows": ["a" * 65], "key": {"a": "g2"}}, self.pal)
        with self.assertRaises(MapError):
            mapkit.art_image({"rows": ["a" * 16] * 16, "key": {"a": "nope"}}, self.pal)
        with self.assertRaises(MapError):
            mapkit.recipe_art({"ops": [{"op": "blob", "cx": 8, "cy": 8, "rx": 0, "ry": 3, "ramp": "stone"}]}, self.pal)
        with self.assertRaises(MapError):
            mapkit.recipe_art({"size": [99, 4], "fill": "g2"}, self.pal)

    def test_outline_and_shift(self):
        dot = {"rows": ["." * 16] * 7 + ["." * 7 + "a" + "." * 8] + ["." * 16] * 8, "key": {"a": "r1"}}
        out = mapkit.art_op("outline", dot, {}, self.pal)
        self.assertEqual(sum(r.count(next(k for k, v in out["key"].items() if v == "n0")) for r in out["rows"]), 4)
        moved = mapkit.art_op("shift", dot, {"dx": 9, "dy": 0}, self.pal)  # wraps past the right edge
        self.assertEqual(moved["rows"][7].index("a"), 0)

    def test_save_tile_validates_and_merges(self):
        with self.assertRaises(MapError):
            self.p.save_tile({"id": "Bad Id", "look": SAND})
        with self.assertRaises(MapError):
            self.p.save_tile({"id": "ash", "layer": "sky", "look": SAND})
        with self.assertRaises(MapError):
            self.p.save_tile({"id": "ash", "layer": "ground"})  # a material needs a look
        with self.assertRaises(MapError):
            self.p.save_tile({"id": "ash", "layer": "ground", "look": {"base": "nope"}})
        with self.assertRaises(MapError):
            self.p.save_tile({"id": "ash", "layer": "ground", "look": {**SAND, "decor": [{"tile": "grass"}]}})
        with self.assertRaises(MapError):
            self.p.save_tile({"id": "post", "layer": "object"})  # an object needs a picture
        r = self.p.save_tile({"id": "ash", "name": "灰地", "layer": "ground", "look": SAND})
        self.assertEqual(r["tile"]["rank"], 26)  # on top of the twenty-six starter materials
        self.assertEqual(r["tile"]["look"]["rim"], None)
        self.p.save_tile({"id": "ash", "pass": "block"})  # everything else is kept
        t = self.p.tiles()["ash"]
        self.assertEqual((t["name"], t["pass"], t["look"]), ("灰地", "block", r["tile"]["look"]))
        post = self.p.save_tile({"id": "post", "layer": "object", "jitter": 99, "art": {
            "size": [6, 20], "ops": [{"op": "rect", "x": 2, "y": 1, "w": 2, "h": 18, "color": "w2"}], "outline": "n0"}})
        self.assertEqual((post["problems"], post["tile"]["jitter"], len(post["tile"]["variants"])), ([], 6, 1))
        self.p.save_tile({"id": "post", "shadow": True})
        self.assertEqual(self.p.tiles()["post"]["variants"], post["tile"]["variants"])


class Maps(Base):
    def test_file_round_trip(self):
        m = self.p.create_map("a", 5, 4, fill="grass")
        self.p.edit("a", lambda m, t: mapkit.set_cells(m, t, [["object", 1, 2, "tree"], ["ground", 0, 0, None]]))
        again = self.p.load_map("a")
        self.assertEqual(again.layers["object"][2][1], "tree")
        self.assertIsNone(again.layers["ground"][0][0])
        self.assertEqual(again.layers["ground"][3][4], "grass")
        raw = json.loads((self.p.data / "a.map.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["legend"], ["grass", "tree"])
        self.assertEqual(m.width, raw["width"])

    def test_set_cells_is_all_or_nothing(self):
        m = Map("x", 3, 3)
        with self.assertRaises(MapError):
            mapkit.set_cells(m, self.tiles, [["ground", 0, 0, "grass"], ["ground", 9, 9, "grass"]])
        with self.assertRaises(MapError):
            mapkit.set_cells(m, self.tiles, [["ground", 0, 0, "tree"]])  # wrong layer
        with self.assertRaises(MapError):
            mapkit.set_cells(m, self.tiles, [["ground", 0, 0, "lava"]])
        self.assertEqual(m.used(), {})

    def test_shapes(self):
        m = Map("x", 8, 6)
        self.assertEqual(mapkit.fill_rect(m, self.tiles, "grass", -2, -2, 100, 100), 48)  # clipped
        self.assertEqual(mapkit.fill_rect(m, self.tiles, "water", 3, 0, 1, 6), 6)
        self.assertEqual(mapkit.flood_fill(m, self.tiles, "forest", 0, 0), 18)  # stops at the river
        self.assertEqual(m.layers["ground"][0][4], "grass")
        self.assertEqual(mapkit.draw_line(m, self.tiles, "mountain", 0, 0, 7, 5), 8)
        self.assertEqual(mapkit.fill_rect(m, self.tiles, None, 0, 0, 8, 1, layer="ground"), 8)
        self.assertEqual(mapkit.line_cells(0, 0, 3, 0), [(0, 0), (1, 0), (2, 0), (3, 0)])

    def test_scatter_respects_ground_spacing_and_seed(self):
        def build(seed_):
            m = Map("x", 12, 12)
            mapkit.fill_rect(m, self.tiles, "grass", 0, 0, 12, 12)
            mapkit.fill_rect(m, self.tiles, "water", 0, 0, 6, 12)
            return m, mapkit.scatter(m, self.tiles, "rock", 5, seed_, spacing=1)

        (m, placed), (_, same), (_, other) = build(1), build(1), build(2)
        self.assertEqual(placed, same)
        self.assertNotEqual(placed, other)
        self.assertEqual(len(placed), 5)
        self.assertTrue(all(x >= 6 for x, _ in placed))  # never on water
        for i, (ax, ay) in enumerate(placed):
            for bx, by in placed[i + 1:]:
                self.assertTrue(max(abs(ax - bx), abs(ay - by)) > 1)
        with self.assertRaises(MapError):
            mapkit.scatter(m, self.tiles, "grass", 3)
        # An islet across the water: `reachable` keeps to the mainland.
        mapkit.fill_rect(m, self.tiles, "grass", 0, 0, 2, 2)
        mapkit.fill_rect(m, self.tiles, None, 0, 0, 12, 12, layer="object")
        self.assertTrue(any(x < 2 for x, _ in mapkit.scatter(m, self.tiles, "rock", 200)))
        mapkit.fill_rect(m, self.tiles, None, 0, 0, 12, 12, layer="object")
        placed = mapkit.scatter(m, self.tiles, "rock", 200, reachable=True)
        self.assertEqual(len(placed), 72)
        self.assertTrue(all(x >= 6 for x, _ in placed))

    def test_resize_moves_content(self):
        m = Map("x", 2, 2)
        mapkit.set_cells(m, self.tiles, [["ground", 0, 0, "water"], ["object", 1, 1, "rock"]])
        big = mapkit.resize(m, 4, 4, dx=1, dy=1, fill="grass", tiles=self.tiles)
        self.assertEqual(big.layers["ground"][1][1], "water")
        self.assertEqual(big.layers["object"][2][2], "rock")
        self.assertEqual(big.layers["ground"][0][0], "grass")
        self.assertIsNone(big.layers["ground"][2][2])  # was empty before, stays empty
        small = mapkit.resize(big, 2, 2, dx=-1, dy=-1)
        self.assertEqual(small.layers, m.layers)

    def test_validate(self):
        m = Map("x", 7, 3)
        mapkit.fill_rect(m, self.tiles, "grass", 0, 0, 7, 3)
        self.assertTrue(mapkit.validate(m, self.tiles)["ok"])
        mapkit.fill_rect(m, self.tiles, "water", 3, 0, 1, 3)
        mapkit.set_cells(m, self.tiles, [["object", 6, 1, "rock"], ["object", 1, 1, "rock"], ["ground", 0, 0, None]])
        m.layers["ground"][2][0] = "lava"
        result = mapkit.validate(m, self.tiles)
        text = "\n".join(result["problems"])
        self.assertIn("'lava'", text)
        self.assertIn("1 格没有地面", text)
        self.assertIn("分成 2 块", result["notes"][0])
        self.assertIn("1 个物件从最大的陆地走不到，例如 rock@(1,1)", text)
        # A blocking object splits the land just like water does - over its whole footprint.
        n = Map("y", 7, 2)
        mapkit.fill_rect(n, self.tiles, "grass", 0, 0, 7, 2)
        mapkit.set_cells(n, self.tiles, [["object", 3, 1, "hut"]])  # 3 wide, 2 deep
        self.assertEqual(mapkit.footprint_cells(self.tiles["hut"], 3, 1), [(2, 0), (3, 0), (4, 0), (2, 1), (3, 1), (4, 1)])
        stats = mapkit.validate(n, self.tiles)["stats"]
        self.assertEqual((stats["walkable_regions"], stats["largest_region"]), (2, 4))
        # Scattering keeps off that ground, and a wide thing needs room for all of itself.
        placed = mapkit.scatter(n, self.tiles, "rock", 99)
        self.assertEqual(sorted(placed), [(0, 0), (0, 1), (1, 0), (1, 1), (5, 0), (5, 1), (6, 0), (6, 1)])
        mapkit.fill_rect(n, self.tiles, None, 0, 0, 7, 2, layer="object")
        huts = mapkit.scatter(n, self.tiles, "hut", 99)
        self.assertTrue(1 <= len(huts) <= 2 and all(y == 1 and 1 <= x <= 5 for x, y in huts), huts)
        covered = [c for x, y in huts for c in mapkit.footprint_cells(self.tiles["hut"], x, y)]
        self.assertEqual(len(covered), len(set(covered)))

    def test_text_view(self):
        m = Map("x", 4, 2)
        mapkit.fill_rect(m, self.tiles, "grass", 0, 0, 4, 2)
        mapkit.set_cells(m, self.tiles, [["object", 2, 1, "tree"], ["ground", 3, 0, "water"]])
        chars = mapkit.tile_chars(self.tiles)
        g, w, t = chars["grass"], chars["water"], chars["tree"]
        text = mapkit.text_view(m, self.tiles)
        self.assertIn(f"  0 {g}{g}{g}{w}", text)
        self.assertIn(f"  1 {g}{g}{t}{g}", text)
        self.assertIn(f"{g}=grass", text)
        self.assertIn(f"  1 {g}{g}{g}{g}", mapkit.text_view(m, self.tiles, layer="ground"))


class Picture(Base):
    """The renderer: not a grid of stamps, but still a pure function of the map."""

    def island(self):
        m = Map("x", 14, 12)
        mapkit.fill_rect(m, self.tiles, "water", 0, 0, 14, 12)
        mapkit.fill_rect(m, self.tiles, "grass", 3, 3, 8, 6)
        mapkit.fill_rect(m, self.tiles, "mountain", 5, 4, 3, 2)
        mapkit.set_cells(m, self.tiles, [["object", 9, 7, "tree"], ["object", 4, 7, "rock"]])
        return m

    def colours(self, img):
        return {c for _, c in img.getcolors(100000)}

    def test_size_scale_and_palette(self):
        m = self.island()
        img = mapkit.render(m, self.tiles, self.pal, scale=2)
        self.assertEqual(img.size, (14 * 16 * 2, 12 * 16 * 2))
        self.assertEqual(mapkit.render(m, self.tiles, self.pal, scale=3, region=(1, 0, 2, 1)).size, (96, 48))
        allowed = {v for k, v in self.pal.items() if not k.startswith("_")}
        self.assertTrue(self.colours(img) <= allowed)  # hard pixels, palette only
        with self.assertRaises(MapError):
            mapkit.render(m, self.tiles, self.pal, region=(99, 99, 2, 2))

    def test_a_region_is_a_crop_of_the_whole(self):
        m = self.island()
        whole = mapkit.render(m, self.tiles, self.pal, scale=1)
        for x, y, w, h in [(0, 0, 5, 5), (4, 3, 6, 6), (8, 5, 6, 7), (9, 7, 1, 1)]:
            part = mapkit.render(m, self.tiles, self.pal, scale=1, region=(x, y, w, h))
            crop = whole.crop((x * 16, y * 16, (x + w) * 16, (y + h) * 16))
            self.assertEqual(part.tobytes(), crop.tobytes(), (x, y, w, h))

    def test_the_ground_is_not_a_grid_of_stamps(self):
        m = Map("x", 8, 8)
        mapkit.fill_rect(m, self.tiles, "grass", 0, 0, 8, 8)
        img = mapkit.render(m, self.tiles, self.pal, scale=1)
        cells = {img.crop((x * 16, y * 16, x * 16 + 16, y * 16 + 16)).tobytes() for x in range(8) for y in range(8)}
        self.assertEqual(len(cells), 64)  # no two cells of the same material look alike
        # A border between two materials is not a straight line down the cell edge.
        mapkit.fill_rect(m, self.tiles, "water", 4, 0, 4, 8)
        px = mapkit.render(m, self.tiles, self.pal, scale=1, objects=False).load()
        water = {self.pal[n][:3] for n in ("b0", "b1", "b2", "b3")}
        edge = {next(x for x in range(128) if px[x, y][:3] in water) for y in range(16, 112)}
        self.assertGreater(len(edge), 3)
        self.assertTrue(all(56 <= x <= 72 for x in edge))  # but it stays near the cell edge

    def test_cells_keep_their_material_at_the_centre(self):
        # Borders bend, and a ground may spill a few blades or stones over them, but the
        # middle of every cell belongs to the ground that cell really has.
        m = self.island()
        win = render._window(m, self.tiles, self.pal, None)
        code = {t["id"]: i + 1 for i, t in enumerate(win["grounds"])}
        o = render.PAD * 16
        for y in range(12):
            for x in range(14):
                self.assertEqual(int(win["g"]["M"][o + y * 16 + 8, o + x * 16 + 8]), code[m.layers["ground"][y][x]], (x, y))
        # ... and where two grounds only meet (no spill declared onto water), the picture agrees.
        m.layers["object"] = [[None] * 14 for _ in range(12)]
        for t in self.tiles.values():
            if t["layer"] == "ground":
                t["look"]["decor"] = []
        px = mapkit.render(m, self.tiles, self.pal, scale=1).load()
        names = {v[:3]: k for k, v in self.pal.items() if not k.startswith("_")}
        for y in range(12):
            for x in range(14):
                if m.layers["ground"][y][x] == "water":
                    self.assertEqual(names[px[x * 16 + 8, y * 16 + 8][:3]][0], "b", (x, y))

    def test_grounds_meet_through_a_zone_not_a_line(self):
        # Meadow on sand: blades reach out over the sand and thin away; nothing reaches far.
        m = Map("x", 12, 6)
        mapkit.fill_rect(m, self.tiles, "sand", 0, 0, 12, 6)
        mapkit.fill_rect(m, self.tiles, "grass", 6, 0, 6, 6)
        for t in self.tiles.values():
            if t["layer"] == "ground":
                t["look"]["decor"] = []
        win = render._window(m, self.tiles, self.pal, None)
        o, code = render.PAD * 16, {t["id"]: i + 1 for i, t in enumerate(win["grounds"])}
        M = win["g"]["M"][o:o + 96, o:o + 192]
        px = mapkit.render(m, self.tiles, self.pal, scale=1).load()
        green = {self.pal[n][:3] for n in ("g1", "g2", "g3", "g4")}
        near = far = 0
        for y in range(96):
            border = next(x for x in range(192) if M[y, x] == code["grass"])
            near += sum(px[x, y][:3] in green for x in range(border - 6, border))
            far += sum(px[x, y][:3] in green for x in range(0, border - 14))
        self.assertGreater(near, 96)   # more than one blade a row in the last few pixels of sand
        self.assertEqual(far, 0)       # and none out in the open sand
        # The spill can be kept off a ground: meadow does not thin onto water.
        mapkit.fill_rect(m, self.tiles, "water", 0, 0, 6, 6)
        win = render._window(m, self.tiles, self.pal, None)
        M = win["g"]["M"][o:o + 96, o:o + 192]
        px = mapkit.render(m, self.tiles, self.pal, scale=1).load()
        bank = {self.pal[n][:3] for n in ("w0", "w1", "w2")}
        for y in range(96):
            for x in range(192):
                if M[y, x] == code["water"]:
                    self.assertTrue(px[x, y][:3] not in green or px[x, y][:3] in bank, (x, y))

    def test_objects_overlap_and_cast_shadows(self):
        m = Map("x", 6, 6)
        mapkit.fill_rect(m, self.tiles, "sand", 0, 0, 6, 6)
        bare = mapkit.render(m, self.tiles, self.pal, scale=1)
        mapkit.set_cells(m, self.tiles, [["object", 3, 4, "tree"]])
        img = mapkit.render(m, self.tiles, self.pal, scale=1)
        changed = [(x, y) for x in range(96) for y in range(96) if bare.getpixel((x, y)) != img.getpixel((x, y))]
        ys = [y for _, y in changed]
        self.assertLess(min(ys), 3 * 16)       # the canopy reaches at least a cell above its own
        self.assertLess(max(ys), 5 * 16 + 13)  # below its feet only its shadow, falling away from the sun
        self.assertEqual(mapkit.render(m, self.tiles, self.pal, scale=1, objects=False).tobytes(), bare.tobytes())

    def test_look_sample_and_preview(self):
        self.assertEqual(render.look_sample(self.tiles["water"], self.tiles, self.pal, scale=1).size, (144, 144))
        self.assertGreater(mapkit.tile_preview(self.tiles["tree"], self.pal, self.tiles).width, 200)
        look = render.clean_look({"base": "g2", "cliff": {"colors": ["w1"], "over": ["water"]}}, self.pal, self.tiles)
        self.assertEqual(look["cliff"], {"colors": ["w1"], "over": ["water"], "cracks": False})


class Motion(Base):
    """The picture is one frame of a loop. What moves is declared, and stays a pure function."""

    def lake(self):
        m = Map("x", 12, 10)
        mapkit.fill_rect(m, self.tiles, "grass", 0, 0, 12, 10)
        mapkit.fill_rect(m, self.tiles, "water", 5, 3, 6, 5)
        mapkit.set_cells(m, self.tiles, [["object", 2, 6, "tree"], ["object", 2, 8, "rock"], ["object", 3, 2, "hut"]])
        return m

    def test_frames_differ_and_the_loop_closes(self):
        m = self.lake()
        frames = render.render_frames(m, self.tiles, self.pal, range(render.FRAMES + 1), scale=1)
        self.assertGreaterEqual(len({f.tobytes() for f in frames[:render.FRAMES]}), render.CYCLE)
        self.assertEqual(frames[0].tobytes(), frames[render.FRAMES].tobytes())
        self.assertEqual(frames[3].tobytes(), mapkit.render(m, self.tiles, self.pal, scale=1, frame=3).tobytes())
        allowed = {v for k, v in self.pal.items() if not k.startswith("_")}
        for f in frames:
            self.assertTrue({c for _, c in f.getcolors(100000)} <= allowed)

    def test_a_region_of_a_frame_is_a_crop_of_that_frame(self):
        m = self.lake()
        whole = mapkit.render(m, self.tiles, self.pal, scale=1, frame=5)
        part = mapkit.render(m, self.tiles, self.pal, scale=1, frame=5, region=(4, 2, 6, 6))
        self.assertEqual(part.tobytes(), whole.crop((64, 32, 160, 128)).tobytes())

    def test_only_what_is_declared_moves(self):
        m = self.lake()
        a, b = (mapkit.render(m, self.tiles, self.pal, scale=1, frame=f) for f in (0, 2))
        moved = {(x // 16, y // 16) for x in range(192) for y in range(160) if a.getpixel((x, y)) != b.getpixel((x, y))}
        self.assertTrue(any(5 <= x <= 10 and 3 <= y <= 7 for x, y in moved))   # the water
        self.assertTrue(any(x in (1, 2, 3) and y in (4, 5) for x, y in moved))  # the tree's crown, above its cell
        # The house does not move, and neither does bare ground with nothing declared on it.
        mapkit.fill_rect(m, self.tiles, "sand", 0, 0, 12, 10)
        m.layers["object"] = [[None] * 12 for _ in range(10)]
        mapkit.set_cells(m, self.tiles, [["object", 5, 5, "hut"]])
        still = render.render_frames(m, self.tiles, self.pal, range(render.FRAMES), scale=1)
        self.assertEqual(len({f.tobytes() for f in still}), 1)

    def test_motion_keeps_the_sprite_and_its_feet(self):
        P = render.Palette(self.pal)
        tree = P.sprite(self.tiles["tree"]["variants"][0])
        h = tree.shape[0]
        for f in range(render.FRAMES):
            sway = render.moved(tree, "sway", f)
            self.assertEqual(sway.shape, tree.shape)
            self.assertTrue((sway[round(h * 0.55):] == tree[round(h * 0.55):]).all())  # the trunk stays put
            self.assertLessEqual(abs(int((sway >= 0).sum()) - int((tree >= 0).sum())), 4)
        self.assertFalse((render.moved(tree, "sway", 2) == tree).all())
        self.assertIs(render.moved(tree, "none", 3), tree)
        person = P.sprite(self.tiles["person"]["variants"][0])
        dip = render.moved(person, "bob", render.FRAMES - 1)
        self.assertTrue((dip[-2:] == person[-2:]).all())
        self.assertFalse((dip == person).all())

    def test_motion_and_anim_are_validated(self):
        with self.assertRaises(MapError):
            self.p.save_tile({"id": "tree", "motion": "dance"})
        with self.assertRaises(MapError):
            self.p.save_tile({"id": "water", "look": {"base": "b1", "specks": [{"color": "b2", "anim": "boil"}]}})
        self.assertEqual(self.p.save_tile({"id": "rock", "motion": "bob"})["tile"]["motion"], "bob")
        self.assertEqual(self.tiles["water"]["look"]["shore"]["lap"], True)
        self.assertEqual(self.tiles["hut"]["motion"], "none")

    def test_wind_comes_in_gusts(self):
        # A meadow is still, then a band of pale streaks crosses it, then it is still again.
        m = Map("x", 30, 6)
        mapkit.fill_rect(m, self.tiles, "grass", 0, 0, 30, 6)
        frames = render.render_frames(m, self.tiles, self.pal, range(render.FRAMES + 1), scale=1)
        self.assertEqual(frames[0].tobytes(), frames[render.FRAMES].tobytes())  # the loop closes
        self.assertEqual(len({f.tobytes() for f in frames[:render.FRAMES]}), render.FRAMES)
        pale = self.pal["g4"][:3]

        def columns(img):  # where along the map the pale streaks are
            px = img.load()
            return [sum(px[x, y][:3] == pale for y in range(96)) for x in range(128)]

        still = min(sum(columns(f)) for f in frames)
        blown = [columns(f) for f in frames[:render.FRAMES]]
        centres = []
        for c in blown:
            extra = [max(0, v - 1) for v in c]
            if sum(extra) > 40:
                centres.append(sum(x * v for x, v in enumerate(extra)) / sum(extra))
        self.assertGreater(len(centres), 4)                       # the gust is seen over several frames
        self.assertLess(len(centres), render.FRAMES)              # and there are frames without one
        self.assertGreater(max(sum(c) for c in blown), still * 2 + 50)
        # Things that the wind pushes over lean only while it is on them.
        tuft = self.tiles["tuft"]
        self.assertEqual(tuft["motion"], "gust")
        spr = render.Palette(self.pal).sprite(tuft["variants"][0])
        self.assertTrue((render.moved(spr, "gust", 0) == spr).all())
        self.assertFalse((render.moved(spr, "gust", 1) == spr).all())
        self.assertEqual(float(render.gust_at(0.5, 0)), 0.0)
        self.assertGreater(float(render.gust_at(render.GUST_WIDTH * 0.9, 0)), 0.8)

    def test_surf_runs_up_the_beach_and_back(self):
        m = Map("x", 10, 10)
        mapkit.fill_rect(m, self.tiles, "sand", 0, 0, 10, 10)
        mapkit.fill_rect(m, self.tiles, "sea", 0, 6, 10, 4)
        frames = render.render_frames(m, self.tiles, self.pal, range(render.FRAMES + 1), scale=1)
        self.assertEqual(frames[0].tobytes(), frames[render.FRAMES].tobytes())
        win = render._window(m, self.tiles, self.pal, None)
        o, code = render.PAD * 16, {t["id"]: i + 1 for i, t in enumerate(win["grounds"])}
        sand = win["g"]["M"][o:o + 160, o:o + 160] == code["sand"]
        wet_colours = {self.pal[n][:3] for n in ("b2", "f0")}

        def flooded(img):  # sand pixels that are under water or foam right now
            px = img.load()
            return sum(1 for y in range(160) for x in range(160) if sand[y, x] and px[x, y][:3] in wet_colours and y > 60)

        amounts = [flooded(f) for f in frames[:render.FRAMES]]
        self.assertGreater(max(amounts), 300)             # the wash covers a real strip of beach
        self.assertLess(min(amounts), max(amounts) // 4)  # and then it is nearly all gone
        # It never climbs past its reach, and never onto ground it was not told to.
        reach = self.tiles["sea"]["look"]["surf"]["reach"]
        top = min(y for y in range(160) if not sand[y].all())
        for f in frames[:render.FRAMES]:
            px = f.load()
            for y in range(0, top - reach - 12):
                for x in range(0, 160, 3):
                    self.assertNotIn(px[x, y][:3], {self.pal["b2"][:3]}, (x, y))
        mapkit.fill_rect(m, self.tiles, "grass", 0, 0, 10, 6)
        still = render.render_frames(m, self.tiles, self.pal, [0, 9, 14, 18], scale=1, region=(0, 0, 10, 4))
        sea_blue = self.pal["b2"][:3]
        self.assertTrue(all(sea_blue not in {c[:3] for _, c in f.getcolors(100000)} for f in still))

    def test_a_river_mouth_is_not_a_shore(self):
        # Where the river runs into the sea, neither does what it does at land: no shallows
        # band, no foam coming in, no lapping edge. Take away `joins` and all of that appears.
        m = Map("x", 12, 10)
        mapkit.fill_rect(m, self.tiles, "sea", 0, 0, 12, 10)
        mapkit.fill_rect(m, self.tiles, "water", 0, 0, 12, 5)
        for t in self.tiles.values():
            if t["layer"] == "ground":
                t["look"]["decor"] = []
        foam, pale = self.pal["f0"][:3], self.pal["b3"][:3]

        def seen(tiles):
            out = set()
            for f in render.render_frames(m, tiles, self.pal, range(0, render.FRAMES, 3), scale=1):
                out |= {c[:3] for _, c in f.getcolors(100000)}
            return out

        joined = seen(self.tiles)
        self.assertNotIn(foam, joined)
        self.assertEqual(self.tiles["water"]["look"]["joins"], ["sea"])
        import copy
        apart = copy.deepcopy(self.tiles)
        apart["water"]["look"]["joins"] = []
        self.assertIn(foam, seen(apart))
        # The same river still has its lapping edge against sand.
        mapkit.fill_rect(m, self.tiles, "sand", 0, 0, 12, 2)
        px = mapkit.render(m, self.tiles, self.pal, scale=1, frame=0).load()
        self.assertTrue(any(px[x, y][:3] == pale for x in range(192) for y in range(24, 44)))

    def test_a_river_runs_to_the_sea(self):
        m = Map("x", 13, 14)
        mapkit.fill_rect(m, self.tiles, "grass", 0, 0, 13, 14)
        mapkit.fill_rect(m, self.tiles, "sand", 0, 7, 13, 2)
        mapkit.fill_rect(m, self.tiles, "sea", 0, 9, 13, 5)
        mapkit.fill_rect(m, self.tiles, "water", 2, 2, 5, 1)   # along, then
        mapkit.fill_rect(m, self.tiles, "water", 6, 2, 1, 7)   # down to the sea
        mapkit.fill_rect(m, self.tiles, "water", 10, 1, 2, 2)  # and a pond that reaches nothing
        fx, fy = render.currents(m, self.tiles)
        for y in range(4, 9):
            self.assertGreater(fy[y, 6], 0.8, y)               # down the last reach
        for x in range(2, 5):
            self.assertGreater(fx[2, x], 0.8, x)               # and along the first, toward the bend
        self.assertEqual((float(fx[1, 10]), float(fy[2, 11])), (0.0, 0.0))  # the pond lies still
        self.assertGreater(fy[9, 6], 0.5)                      # off the mouth it carries on...
        self.assertGreater(fy[9, 6], fy[11, 6])                # ...weakening
        self.assertEqual(float(fy[13, 0]), 0.0)                # and the open sea has none
        self.assertIsNone(render.currents(self.island_free(), self.tiles))
        # A piece of the picture still agrees with the whole, at any frame.
        whole = mapkit.render(m, self.tiles, self.pal, scale=1, frame=5)
        part = mapkit.render(m, self.tiles, self.pal, scale=1, frame=5, region=(4, 5, 6, 7))
        self.assertEqual(part.tobytes(), whole.crop((64, 80, 160, 192)).tobytes())
        # The ripples travel: what marks the river at one frame is further down it at the next.
        for t in self.tiles.values():
            if t["layer"] == "ground":
                t["look"]["decor"] = []
        ripple = self.pal["b2"][:3]

        def marks(frame):  # ripple pixels in the straight reach, well inside its banks
            px = mapkit.render(m, self.tiles, self.pal, scale=1, frame=frame).load()
            return {(x, y) for x in range(100, 108) for y in range(56, 104) if px[x, y][:3] == ripple}

        a, b = marks(2), marks(3)
        self.assertTrue(a and b)
        moved_down = len(a & {(x, y - 1) for x, y in b}) + len(a & {(x, y - 2) for x, y in b})
        moved_up = len(a & {(x, y + 1) for x, y in b}) + len(a & {(x, y + 2) for x, y in b})
        self.assertGreater(moved_down, moved_up)
        # The sea washes the beach - but not beside the mouth, where the river is pushing out.
        wash = {self.pal["b2"][:3]}  # the colour of water run up over sand
        win = render._window(m, self.tiles, self.pal, None)
        o, code = render.PAD * 16, {t["id"]: i + 1 for i, t in enumerate(win["grounds"])}
        sand = win["g"]["M"][o:o + 224, o:o + 208] == code["sand"]  # borders wander: ask, do not assume
        beside = far = 0
        for f in render.render_frames(m, self.tiles, self.pal, range(render.FRAMES), scale=1):
            px = f.load()
            beside += sum(px[x, y][:3] in wash for x in list(range(84, 94)) + list(range(114, 124))
                          for y in range(128, 142) if sand[y, x])
            far += sum(px[x, y][:3] in wash for x in range(4, 24) for y in range(128, 142) if sand[y, x])
        self.assertGreater(far, 200)
        self.assertLess(beside, far // 6)

    def island_free(self):
        m = Map("x", 4, 4)
        mapkit.fill_rect(m, self.tiles, "grass", 0, 0, 4, 4)
        return m

class Plants(Base):
    """What grows on the ground: kept per cell on its own layer, drawn as a cover, not as sprites."""

    def test_every_form_grows(self):
        # each way of growing draws something, and only the crust stays put in a gust
        for pid in ("fernbed", "heather", "lichen", "reeds"):
            m = Map("f", 5, 5)
            mapkit.fill_rect(m, self.tiles, "heath" if pid != "lichen" else "mountain", 0, 0, 5, 5)
            bare = np.asarray(mapkit.render(m, self.tiles, self.pal, scale=1)).copy()
            mapkit.set_cells(m, self.tiles, [["plant", x, y, pid] for x in range(1, 4) for y in range(1, 4)])
            frames = [np.asarray(f) for f in render.render_frames(m, self.tiles, self.pal, range(render.FRAMES), scale=1)]
            self.assertTrue((frames[0] != bare).any(), pid)
        self.assertEqual(self.tiles["lichen"]["grow"]["form"], "crust")
        self.assertTrue(self.tiles["lichen"]["grow"]["still"] and self.tiles["reeds"]["grow"]["stiff"])

    def test_long_grass_grows_past_its_cell(self):
        m = Map("g", 5, 5)
        mapkit.fill_rect(m, self.tiles, "grass", 0, 0, 5, 5)
        bare = mapkit.render(m, self.tiles, self.pal, scale=1)
        mapkit.set_cells(m, self.tiles, [["plant", 2, 2, "longgrass"]])
        grown = mapkit.render(m, self.tiles, self.pal, scale=1)
        diff = np.asarray(bare).astype(int) - np.asarray(grown).astype(int)
        ys = np.nonzero(np.abs(diff).sum(-1).any(1))[0]
        self.assertLess(ys.min(), 2 * mapkit.CELL)          # blades stand up into the cell above
        self.assertNotIn("longgrass", {d["tile"] for d in render.instances(m, self.tiles, self.pal)})   # it is no object
        again = Map.from_json(json.loads(json.dumps(m.to_json())))
        self.assertEqual(again.layers["plant"][2][2], "longgrass")
        with self.assertRaises(MapError):
            mapkit.set_cells(m, self.tiles, [["object", 1, 1, "longgrass"]])
        with self.assertRaises(MapError):
            self.p.save_tile({"id": "weed", "layer": "plant", "grow": {"height": [2, 40], "colors": ["g1", "g2"]}})


class Disk(Base):
    def test_stale_edit_is_reported(self):
        self.p.create_map("a", 3, 3, fill="grass")
        rev = self.p.map_rev("a")
        first = self.p.edit("a", lambda m, t: mapkit.fill_rect(m, t, "water", 0, 0, 1, 1), base_rev=rev)
        self.assertFalse(first["stale"])
        self.assertEqual(first["result"], 1)
        second = self.p.edit("a", lambda m, t: mapkit.fill_rect(m, t, "water", 1, 0, 1, 1), base_rev=rev)
        self.assertTrue(second["stale"])

    def test_delete_tile_guard_and_force(self):
        self.p.create_map("a", 3, 3, fill="grass")
        with self.assertRaises(MapError):
            self.p.delete_tile("grass")
        with self.assertRaises(MapError):
            self.p.save_tile({"id": "grass", "layer": "object", "art": {"fill": "g2"}})
        with self.assertRaises(MapError):
            self.p.delete_tile("tuft")  # grass sprinkles it
        self.p.delete_tile("tuft", force=True)
        self.assertNotIn("tuft", [d["tile"] for d in self.p.tiles()["grass"]["look"]["decor"]])
        self.assertEqual(self.p.delete_tile("grass", force=True)["cleared"], {"a": 9})
        self.assertEqual(self.p.load_map("a").used(), {})
        self.assertNotIn("grass", self.p.tiles())

    def test_trash_names_export_and_demo(self):
        seed.seed_demo(self.p)
        demo = self.p.load_map("demo")
        self.assertEqual((demo.width, demo.height), (48, 32))
        self.assertEqual(sorted(demo.used()), ["forest", "grass", "mountain", "sand", "water"])  # ground only, no paths
        check = mapkit.validate(demo, self.p.tiles())
        self.assertEqual((check["problems"], check["notes"]), ([], []))  # the ford joins the two banks
        with self.assertRaises(MapError):
            self.p.create_map("demo", 4, 4)
        with self.assertRaises(MapError):
            self.p.create_map("../evil", 4, 4)
        with self.assertRaises(MapError):
            self.p.create_map("huge", 999, 4)
        out = self.p.export()
        self.assertEqual((out["objects"], out["problems"], len(out["maps"])), (17, [], 1))   # with the five kinds of tree
        index_plants = json.loads(Path(out["json"]).read_text(encoding="utf-8"))["plants"]
        self.assertTrue({"shortgrass", "longgrass", "flowergrass", "heather", "reeds", "lichen", "marram"} <= set(index_plants))
        self.assertIn("grow", index_plants["longgrass"])
        index = json.loads(Path(out["json"]).read_text(encoding="utf-8"))
        self.assertEqual((index["frames"], index["fps"]), (render.FRAMES, render.FPS))
        self.assertEqual((index["grounds"]["sea"]["rank"], index["cycle"], index["wind"]["span"]), (0, render.CYCLE, render.GUST_SPAN))
        self.assertEqual((index["objects"]["hut"]["footprint"], index["objects"]["tree"]["motion"]), ([3, 2], "sway"))
        self.assertEqual([len(v) for v in index["objects"]["tree"]["variants"]], [render.CYCLE] * 3)  # a frame each
        self.assertEqual([len(v) for v in index["objects"]["tuft"]["variants"]], [2, 2])            # standing, leaning
        self.assertEqual([len(v) for v in index["objects"]["hut"]["variants"]], [1])         # it stands still
        with mapkit.Image.open(out["png"]) as sheet:
            for name, tile in self.p.tiles().items():
                if tile["layer"] != "object":
                    continue
                for vi, rects in enumerate(index["objects"][name]["variants"]):
                    for want, (x, y, w, h) in zip(render.sprite_frames(tile, vi, self.pal), rects):
                        self.assertEqual(sheet.crop((x, y, x + w, y + h)).tobytes(), want.tobytes(), name)
        # The ground: a still picture, then for each later frame only what differs from it.
        info, stem = out["maps"][0], Path(out["maps"][0]["ground"]).with_suffix("").with_suffix("")
        self.assertEqual(info["moving_frames"], render.FRAMES - 1)
        frames = render.render_frames(demo, self.p.tiles(), self.pal, range(render.FRAMES), scale=1, objects=False)
        with mapkit.Image.open(info["ground"]) as ground:
            self.assertEqual(ground.size, (768, 512))
            for n in range(1, render.FRAMES):
                with mapkit.Image.open(f"{stem}.ground.{n}.png") as over:
                    whole = ground.convert("RGBA")
                    whole.alpha_composite(over.convert("RGBA"))
                    self.assertEqual(whole.tobytes(), frames[n].tobytes(), n)
                    self.assertLess(sum(1 for a in over.convert("RGBA").getdata(3) if a), 768 * 512 // 6)  # mostly empty
        # Everything standing on it, placed for the game: back to front, with a phase each.
        things = json.loads(Path(f"{stem}.things.json").read_text(encoding="utf-8"))
        self.assertEqual(info["things"], len(things))
        self.assertEqual(sum(1 for t in things if not t["decor"]), sum(n for k, n in demo.used().items()
                                                                       if self.p.tiles()[k]["layer"] == "object"))
        self.assertTrue(any(t["decor"] for t in things))
        self.assertEqual([t["y"] for t in things], sorted(t["y"] for t in things))
        self.assertGreater(len({t["phase"] for t in things if t["tile"] == "tuft"}), 3)
        self.assertTrue(Path(self.p.trash_map("demo")).exists())
        self.assertEqual(self.p.list_maps(), [])


if __name__ == "__main__":
    unittest.main()
