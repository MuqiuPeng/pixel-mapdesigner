"""python -m unittest discover -s tools/mapdesigner"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import geo
import mapkit
import seed
from mapkit import MapError, Project

# The measured valleys are not shipped (dem/fetch.py downloads them); tests that need one skip.
HAVE_DEM = "glen_brittle" in geo.sources()
need_dem = unittest.skipUnless(HAVE_DEM, "真实地形数据未下载（python dem/fetch.py）")


class Valley(unittest.TestCase):
    @need_dem
    def test_there_are_real_valleys_and_where_they_came_from(self):
        S = geo.sources()
        self.assertIn("glen_brittle", S)
        for name, info in S.items():
            self.assertTrue((geo.DEM / f"{name}.npy").exists(), name)
            self.assertIn(info["climate"], geo.CLIMATES)
        self.assertIn("attribution", (geo.DEM / "sources.json").read_text(encoding="utf-8"))

    @need_dem
    def test_real_land_has_a_floor_and_walls(self):
        # What the imitation lacked, measured: a real share of flat ground and of steep.
        V = geo.valley(2, "oceanic", "glen_brittle")
        deg = V["degrees"][V["z"] >= 0]
        self.assertGreater(float((deg < 5).mean()), 0.05)
        self.assertGreater(float((deg > 30).mean()), 0.05)
        sea = V["z"] < 0
        self.assertTrue(0.08 < float(sea.mean()) < 0.25)
        self.assertTrue(sea[-1].any())  # the sea comes in from the south

    @need_dem
    def test_water_runs_downhill_and_gathers(self):
        V = geo.valley(2, "oceanic", "glen_brittle")
        H, W = V["z"].shape
        ys, xs = np.nonzero(V["channel"])
        self.assertGreater(len(ys), 100)
        for y, x in zip(ys.tolist(), xs.tolist()):
            r = int(V["recv"][y, x])
            if r < 0:
                continue
            ry, rx = divmod(r, W)
            self.assertGreaterEqual(V["water"][ry, rx], V["water"][y, x])     # more water below than above
            self.assertTrue(V["channel"][ry, rx] or V["z"][ry, rx] < 0 or ry == H - 1, (x, y))  # a stream does not stop
        self.assertGreater(V["steep"], V["gentle"])

    @need_dem
    def test_climate_changes_what_the_same_land_carries(self):
        wet, dry = geo.valley(2, "oceanic", "glen_brittle"), geo.valley(2, "dry", "glen_brittle")
        self.assertTrue((wet["z"] == dry["z"]).all())                 # the same land
        self.assertGreater(int(wet["channel"].sum()), int(dry["channel"].sum()) * 1.5)   # more streams where it rains
        with self.assertRaises(MapError):
            geo.valley(2, "tropical", "glen_brittle")
        with self.assertRaises(MapError):
            geo.valley(2, "oceanic", "atlantis")

    def test_grown_land_looks_like_real_land(self):
        # The generator is calibrated on the real valleys: the same kind of share of flat floor
        # and of steep wall, and the height spread out the same way.
        def numbers(V):
            land = V["z"] >= 0
            deg, z = V["degrees"][land], V["z"][land]
            return float((deg < 5).mean()), float((deg > 30).mean()), float(z.mean() / np.quantile(z, 0.99))

        # Glen Brittle, measured (see dem/sources.json); the same numbers when it is not downloaded
        real = numbers(geo.valley(2, "oceanic", "glen_brittle")) if HAVE_DEM else (0.127, 0.126, 0.33)
        for s in (2, 5):
            V = geo.valley(s, "oceanic")
            self.assertEqual(V["z"].shape, (geo.VALLEY[1], geo.VALLEY[0]))
            self.assertIn("生成", V["origin"])
            flat, steep, hyps = numbers(V)
            self.assertTrue(0.04 < flat < 0.3, flat)
            self.assertTrue(0.02 < steep < 0.25, steep)
            self.assertLess(abs(hyps - real[2]), 0.15)
            sea = V["z"] < 0
            self.assertTrue(0.08 < float(sea.mean()) < 0.25)
            self.assertTrue(sea[-1].any())
        self.assertFalse((geo.valley(2, "oceanic")["z"] == geo.valley(5, "oceanic")["z"]).all())

    def test_grown_rivers_join_and_reach_the_sea(self):
        V = geo.valley(2, "oceanic")
        H, W = V["z"].shape
        ys, xs = np.nonzero(V["channel"])
        self.assertGreater(len(ys), 100)
        for y, x in zip(ys.tolist(), xs.tolist()):
            r = int(V["recv"][y, x])
            if r >= 0:
                ry, rx = divmod(r, W)
                self.assertTrue(V["channel"][ry, rx] or V["z"][ry, rx] < 0 or ry == H - 1, (x, y))


class Place(unittest.TestCase):
    def cut(self, at, **kw):
        return geo.place(40, 28, 2, kw.get("climate", "oceanic"), at, kw.get("source"))

    def test_same_request_same_place(self):
        a, b = self.cut("river"), self.cut("river")
        self.assertEqual(a["cells"], b["cells"])
        self.assertEqual(a["height"], b["height"])
        self.assertNotEqual(a["cells"], self.cut("mouth")["cells"])
        self.assertEqual(len(a["cells"]), 28)
        self.assertEqual(len(a["cells"][0]), 40)

    def test_a_river_place_has_a_river_and_sorted_banks(self):
        out = self.cut("river")
        flat = [t for row in out["cells"] for t in row]
        self.assertGreater(flat.count("water"), 20)
        self.assertGreater(sum(flat.count(t) for t in ("mud", "sand", "gravel")), 20)
        self.assertNotIn("sea", flat)
        # heights are the whole valley's levels: this is low ground, and nothing here is sea
        levels = [v for row in out["height"] for v in row]
        self.assertTrue(1 <= min(levels) and max(levels) <= geo.LEVELS * geo.FINE)
        self.assertGreater(len(set(levels)), 6)   # fine enough to tell which way the ground falls

    def test_stones_where_it_is_strong_mud_where_it_idles(self):
        # The law the banks are sorted by (stream power), checked over several valleys: where the
        # river is strong for its size there are more stones among what it leaves than where it idles.
        def tally(at):
            out = {"gravel": 0, "sand": 0, "mud": 0}
            for s in (2, 3, 4, 5):
                cells = geo.place(40, 28, s, "oceanic", at)["cells"]
                for t in out:
                    out[t] += sum(row.count(t) for row in cells)
            return out

        strong, idle = tally("gorge"), tally("river")
        share = lambda d: d["gravel"] / max(d["gravel"] + d["mud"], 1)  # noqa: E731
        self.assertGreater(share(strong), share(idle) + 0.1)
        self.assertGreater(idle["mud"] + idle["sand"], idle["gravel"])
        V = geo.valley(2, "oceanic")
        self.assertGreater(V["steep"], V["gentle"])

    def test_the_mouth_meets_the_sea(self):
        out = self.cut("mouth")
        flat = [t for row in out["cells"] for t in row]
        self.assertGreater(flat.count("sea"), 40)
        self.assertGreater(flat.count("water"), 10)
        sea_levels = {out["height"][y][x] for y in range(28) for x in range(40) if out["cells"][y][x] == "sea"}
        self.assertEqual(sea_levels, {0})

    def test_by_coordinates_and_bad_requests(self):
        out = geo.place(40, 28, 2, "oceanic", (60, 40))
        self.assertIn("谷地坐标", out["report"][0])
        with self.assertRaises(MapError):
            self.cut("volcano")
        with self.assertRaises(MapError):
            geo.place(4, 4, 2)


class Designed(unittest.TestCase):
    """The main valley, from designs/main.json: an island, and its coast and tops by their laws."""

    @classmethod
    def setUpClass(cls):
        cls.out = geo.place(0, 0, 1, "oceanic", "whole", None, "main")
        cls.g = np.array(cls.out["cells"])
        cls.lv = np.array(cls.out["height"])

    def test_an_island(self):
        g = self.g
        for edge in (g[0], g[-1], g[:, 0], g[:, -1]):
            self.assertTrue((edge == "sea").all())               # sea on every side
        self.assertEqual(g.shape, (geo.designs()["main"]["grid"][1] * 4, geo.designs()["main"]["grid"][0] * 4))

    def test_the_coast_by_its_laws(self):
        g = self.g
        water = np.isin(g, ["sea", "platform", "pool", "stack", "lagoon"])   # what was sea before the shore was worked
        land = ~water
        for kind in ("platform", "shingle", "cave", "stack"):
            self.assertTrue((g == kind).any(), kind)
            # all of it at the shore: within a few steps of the sea, or (if in the sea) of the land
            near = geo._spread((land | (g == "stack")) if kind == "platform" else land if kind == "stack" else water, 8) <= 8
            self.assertTrue(near[g == kind].all(), kind)
        self.assertTrue((self.lv[g == "platform"] == 1).all())   # the shelf lies at the water
        self.assertGreater(self.lv[g == "stack"].min(), geo.FINE)  # a stack stands up out of it
        pools = g == "pool"                                       # tide pools lie in the shelf
        self.assertTrue((geo._spread(g == "platform", 3)[pools] <= 3).all())
        # the lagoon is shut in behind the spit: none of it touches the open sea but at the inlet
        self.assertTrue((g == "lagoon").any())
        self.assertTrue((g == "saltmarsh").any())
        self.assertLess(self.lv[g == "saltmarsh"].max(), geo.FINE)  # marsh is low ground

    def test_biomes_come_in_regions_and_follow_the_land(self):
        b = np.array(self.out["biome"], dtype=object)
        g = self.g
        self.assertTrue(set(b.ravel()) <= set(geo.BIOMES))
        self.assertTrue((b[g == "sea"] == "sea").all() and (b[g == "snow"] == "snowy_slopes").all())
        for kind in ("forest", "plains", "riverside", "moor"):
            self.assertTrue((b == kind).any(), kind)
        # whole regions, not a speckle: most cells share their biome with all four neighbours
        same = (b[1:-1, 1:-1] == b[:-2, 1:-1]) & (b[1:-1, 1:-1] == b[2:, 1:-1]) & (b[1:-1, 1:-1] == b[1:-1, :-2]) & (b[1:-1, 1:-1] == b[1:-1, 2:])
        self.assertGreater(same.mean(), 0.78)
        # trees: in the woods, by the biome's own kinds; none in the sea or on the snow
        objs = np.array(self.out["objects"], dtype=object)
        self.assertGreater(((objs == "pine") | (objs == "oak") | (objs == "birch") | (objs == "alder")).sum(), 200)
        self.assertFalse(((objs != None) & np.isin(g, ["sea", "snow", "water"])).any())  # noqa: E711
        self.assertGreater((objs[b == "riverside"] == "alder").sum(), (objs[b == "riverside"] == "pine").sum())

    def test_snow_on_the_top_of_the_big_mountain(self):
        lv = self.lv
        self.assertTrue((self.g == "snow").any())
        self.assertGreater(lv[self.g == "snow"].min(), geo.FINE * geo.LEVELS * 0.7)

    def test_bog_on_the_flat_wet_tops(self):
        g, lv = self.g, self.lv
        self.assertTrue((g == "bog").any() and (g == "bogpool").any())
        self.assertGreater(np.median(lv[g == "bog"]), np.median(lv[(g != "sea") & (g != "bog")]))
        self.assertFalse(((g == "bogpool") & (geo._spread(g == "bog", 6) > 6)).any())   # pools only in the bog


class Walkable(unittest.TestCase):
    """A map is no good if it cannot be walked: cliffs only where steep, and ways through them."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = Project(data_dir=Path(self.tmp.name) / "maps")
        seed.seed_tiles(self.p)
        self.tiles = self.p.tiles()

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_piece_of_land_can_be_reached(self):
        cases = ((None, "gorge"), (None, "scarp"), (None, "river"), ("glen_brittle", "gorge"), ("hornvik", "gorge"))
        for source, at in (c for c in cases if c[0] is None or c[0] in geo.sources()):
            climate = geo.sources()[source]["climate"] if source else "oceanic"
            out = geo.make_map(self.p, f"{source}_{at}", 48, 32, 2, climate, at, source)
            m = self.p.load_map(f"{source}_{at}")
            _, sizes, main = mapkit.land_regions(m, self.tiles)
            stranded = [n for i, (n, _) in enumerate(sizes) if i != main and n >= 4]
            self.assertEqual(stranded, [], (source, at, out["connect"]))
            self.assertIn("通行", out["report"][-1])

    def test_cliffs_only_where_the_ground_is_nearly_vertical(self):
        # Steepness is the local gradient, in degrees at the map's scale: a step of the land on
        # a slope one can climb is no cliff, so cliffs are few, and only on the steep places.
        total = {"river": 0, "scarp": 0}
        for s in (2, 3, 4, 5):
            for at in total:
                geo.make_map(self.p, f"{at}{s}", 60, 40, s, "oceanic", at)
                m = self.p.load_map(f"{at}{s}")
                walls = mapkit.cliffs(m, self.tiles)
                total[at] += len(walls)
                self.assertLess(len(walls), 0.03 * 2 * m.width * m.height)       # few, everywhere
                tiers, grade = mapkit.tiers(m), mapkit.gradient(m)
                for x, y, dx, dy in walls:
                    self.assertNotEqual(tiers[y][x], tiers[y + dy][x + dx])   # always a step between two tiers
                    self.assertGreaterEqual((grade[y][x] + grade[y + dy][x + dx]) / 2, mapkit.CLIFF_DEG)
        self.assertGreater(total["scarp"], total["river"])

    def test_the_scale_decides_what_is_steep(self):
        m = mapkit.Map("s", 6, 4)
        m.layers["ground"] = [["grass"] * 6 for _ in range(4)]
        m.relief = [[10, 10, 10, 18, 18, 18] for _ in range(4)]
        m.scale = [12.4, 3.3]            # 8 units = 26 m over 12 m: a face
        self.assertEqual(len(mapkit.cliffs(m, self.tiles)), 4)
        m.scale = [40.0, 1.0]            # the same step as 8 m over 40 m: a slope
        self.assertEqual(mapkit.cliffs(m, self.tiles), set())

    def test_a_ramp_opens_a_cliff_and_survives_saving(self):
        self.p.create_map("g", 12, 8)

        def build(m, tiles):
            m.layers["ground"] = [["grass"] * 12 for _ in range(8)]
            m.relief = [[10] * 6 + [60] * 6 for _ in range(8)]     # a wall down the middle
            return geo.connect(m, tiles)

        self.assertEqual(self.p.edit("g", build)["result"]["ramps"], 1)
        m = self.p.load_map("g")
        self.assertTrue(m.ramps)
        self.assertEqual(m.scale, list(mapkit.SCALE))
        walls = mapkit.cliffs(m, self.tiles)
        for x, y, dx, dy in m.ramps:
            self.assertNotIn((x, y, dx, dy), walls)
        m.ramps = []
        self.assertTrue(mapkit.cliffs(m, self.tiles) > walls)  # take the ramps away and the cliffs close


class Saved(unittest.TestCase):
    def test_a_generated_map_keeps_its_heights(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Project(data_dir=Path(tmp) / "maps", export_dir=Path(tmp) / "out")
            seed.seed_tiles(p)
            out = geo.make_map(p, "glen", 40, 28, 2, "oceanic", "mouth")
            m = p.load_map("glen")
            self.assertEqual(m.relief, out["height"])
            # the ground is what was worked out, but for the fords laid to join the banks
            for y, row in enumerate(m.layers["ground"]):
                for x, g in enumerate(row):
                    self.assertTrue(g == out["cells"][y][x] or (g == "ford" and out["cells"][y][x] == "water"), (x, y))
            self.assertEqual(mapkit.validate(m, p.tiles())["problems"], [])
            # the river on it runs to the sea
            import render
            fx, fy = render.currents(m, p.tiles())
            self.assertGreater(float(np.hypot(fx, fy).max()), 0.5)
            # where there is no sea, a river still knows which way is down: it runs toward lower ground
            geo.make_map(p, "inland", 40, 28, 2, "oceanic", "river")
            inland = p.load_map("inland")
            fx, fy = render.currents(inland, p.tiles())
            falls = rises = 0
            for y in range(28):
                for x in range(40):
                    if inland.layers["ground"][y][x] != "water" or abs(fx[y, x]) + abs(fy[y, x]) < 0.5:
                        continue
                    nx, ny = x + int(round(3 * float(fx[y, x]))), y + int(round(3 * float(fy[y, x])))   # three steps on
                    if 0 <= nx < 40 and 0 <= ny < 28 and inland.layers["ground"][ny][nx] == "water":
                        falls += inland.relief[ny][nx] < inland.relief[y][x]
                        rises += inland.relief[ny][nx] > inland.relief[y][x]
            self.assertGreater(falls, rises * 2)
            # heights survive a resize along with the ground
            p.edit("glen", lambda m, t: mapkit.resize(m, 44, 30, 2, 1, tiles=t))
            grown = p.load_map("glen")
            self.assertEqual(grown.relief[1][2], out["height"][0][0])
            self.assertEqual((len(grown.relief), len(grown.relief[0])), (30, 44))
            with self.assertRaises(MapError):
                geo.make_map(p, "glen", 40, 28, 2)  # the name is taken


if __name__ == "__main__":
    unittest.main()
