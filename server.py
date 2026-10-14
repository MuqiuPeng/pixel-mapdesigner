"""The map designer's web editor.

    python tools/mapdesigner/server.py [--port 8765]

Serves web/ and a small JSON API over mapkit, on 127.0.0.1 only. It keeps no state of its
own: every request reads and writes the project files, so edits made at the same time through
the MCP server appear here (the page polls /api/rev).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import geo
import mapkit
import seed
from mapkit import MapError, Project

WEB = Path(__file__).resolve().parent / "web"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".png": "image/png"}
PROJECT = Project()

# Drawing the map is slow pure-Python-and-numpy work, and the editor asks for it in many
# pieces at once, so pieces are drawn in worker processes (threads would queue on the GIL),
# and what has been drawn is kept until the map or the tiles change.
_POOL = None
_PICTURES, _PICTURE_BYTES, _PICTURE_LIMIT = {}, [0], 600 * 2 ** 20
_WORKER = {}


def _draw(data_dir, name, which, scale, grid, region):
    """A piece of the map as PNG bytes. Runs in a worker process."""
    import render as R
    P = _WORKER.get(data_dir) or _WORKER.setdefault(data_dir, Project(data_dir=Path(data_dir)))
    imgs = R.render_frames(P.load_map(name), P.tiles(), P.palette(), which, scale=scale, grid=grid, region=region)
    return mapkit.png_bytes(R.strip(imgs))


def _picture(name, which, scale, grid, region):
    key = (name, PROJECT.map_rev(name), PROJECT.tileset_rev(), tuple(which), scale, grid, tuple(region or ()))
    if key in _PICTURES:
        _PICTURES[key] = _PICTURES.pop(key)          # most recently used last
        return _PICTURES[key]
    args = (str(PROJECT.data), name, list(which), scale, grid, region)
    png = _POOL.submit(_draw, *args).result() if _POOL else _draw(*args)
    _PICTURES[key] = png
    _PICTURE_BYTES[0] += len(png)
    while _PICTURE_BYTES[0] > _PICTURE_LIMIT and _PICTURES:
        _PICTURE_BYTES[0] -= len(_PICTURES.pop(next(iter(_PICTURES))))
    return png


def _state() -> dict:
    import render as R
    return {"palette": PROJECT.palette_json(), "tileset": PROJECT.tileset(), "maps": PROJECT.list_maps(),
            "rev": PROJECT.revs(), "shapes": list(mapkit.SHAPES), "max_side": mapkit.MAX_SIDE,
            "frames": R.FRAMES, "fps": R.FPS, "motions": list(R.MOTIONS), "speck_anims": list(R.SPECK_ANIMS),
            "geo": {"places": geo.PLACES, "climates": {k: v["name"] for k, v in geo.CLIMATES.items()},
                    "valleys": {k: {"name": v["name"], "climate": v["climate"]} for k, v in geo.sources().items()},
                    "fine": geo.FINE, "biomes": {k: v["name"] for k, v in geo.BIOMES.items()}}}


def _map_payload(name: str) -> dict:
    return {**PROJECT.load_map(name).to_json(), "rev": PROJECT.map_rev(name)}


class Handler(BaseHTTPRequestHandler):
    server_version = "mapdesigner"

    def log_message(self, fmt, *args):  # only failures are worth a line
        if len(args) > 1 and str(args[1]).startswith(("4", "5")):
            sys.stderr.write("%s %s\n" % (args[1], args[0]))

    # -- replies

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code: int = 200) -> None:
        self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self) -> dict:
        # Requiring JSON keeps other sites' plain form posts out: they cannot send this type
        # without a preflight, which is never answered.
        if "application/json" not in self.headers.get("Content-Type", ""):
            raise MapError("请求体必须是 JSON")
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if not isinstance(data, dict):
            raise MapError("请求体应该是一个对象")
        return data

    def _route(self, method: str) -> None:
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.split("/") if p]
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if parts[:1] != ["api"]:
                return self._static(parts) if method == "GET" else self._json({"error": "not found"}, 404)
            handler = getattr(self, f"_{method.lower()}_{parts[1] if len(parts) > 1 else ''}", None)
            if handler is None:
                return self._json({"error": "not found"}, 404)
            handler(parts[2:], query)
        except MapError as e:
            self._json({"error": str(e)}, 400)
        except (ValueError, KeyError, TypeError) as e:
            self._json({"error": f"请求有误：{e!r}"}, 400)

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_DELETE(self):
        self._route("DELETE")

    # -- static files

    def _static(self, parts: list) -> None:
        # /preview/... shows what is under art/preview (design sheets, trial renders).
        base = mapkit.ROOT / "art" / "preview" if parts[:1] == ["preview"] else WEB
        rel = "/".join(parts[1:] if base != WEB else parts) or "index.html"
        path = (base / rel).resolve()
        if base not in path.parents or not path.is_file():
            return self._json({"error": "not found"}, 404)
        self._send(200, path.read_bytes(), TYPES.get(path.suffix, "application/octet-stream"))

    # -- GET

    def _get_state(self, rest, query):
        self._json(_state())

    def _get_rev(self, rest, query):
        self._json(PROJECT.revs())

    def _get_map(self, rest, query):
        name = rest[0]
        if rest[1:] == ["render.png"]:
            # With x/y/w/h (cells) only that part is drawn; it matches the whole pixel for pixel.
            # `frames=all` gives every frame of the loop stacked top to bottom in one picture.
            import render as R
            region = [int(query[k]) for k in "xywh"] if all(k in query for k in "xywh") else None
            which = range(R.FRAMES) if query.get("frames") == "all" else [int(query.get("frame", 0))]
            png = _picture(name, which, int(query.get("scale", 2)), query.get("grid") == "1", region)
            return self._send(200, png, "image/png")
        self._json(_map_payload(name))

    def _get_swatch(self, rest, query):
        """A ground tile as a small square of its surface, for the tile list."""
        tiles = PROJECT.tiles()
        tile = mapkit._tile_for(tiles, rest[0].removesuffix(".png"))
        m = mapkit.Map("swatch", 2, 2)
        if tile["layer"] == "plant":   # what grows, on meadow
            under = "grass" if "grass" in tiles else next(t["id"] for t in tiles.values() if t["layer"] == "ground")
            m.layers["ground"] = [[under] * 2 for _ in range(2)]
            m.layers["plant"] = [[tile["id"]] * 2 for _ in range(2)]
        else:
            m.layers["ground"] = [[tile["id"]] * 2 for _ in range(2)]
        self._send(200, mapkit.png_bytes(mapkit.render(m, tiles, PROJECT.palette(), scale=1)), "image/png")

    # -- POST

    def _post_map(self, rest, query):
        body = self._body()
        if not rest:  # create
            m = PROJECT.create_map(body["name"], int(body["width"]), int(body["height"]), body.get("fill") or None)
            return self._json(_map_payload(m.name))
        name, action = rest[0], rest[1] if len(rest) > 1 else ""
        if action == "patch":
            r = PROJECT.edit(name, lambda m, tiles: mapkit.set_cells(m, tiles, body["cells"]), body.get("base_rev"))
            return self._json({"rev": r["rev"], "stale": r["stale"], "changed": r["result"]})
        if action == "resize":
            PROJECT.edit(name, lambda m, tiles: mapkit.resize(
                m, int(body["width"]), int(body["height"]), int(body.get("dx", 0)), int(body.get("dy", 0)),
                body.get("fill") or None, tiles))
            return self._json(_map_payload(name))
        if action == "validate":
            return self._json(mapkit.validate(PROJECT.load_map(name), PROJECT.tiles()))
        self._json({"error": "not found"}, 404)

    def _post_generate(self, rest, query):
        """A map worked out by the laws in geo.py, saved under a new name."""
        body = self._body()
        out = geo.make_map(PROJECT, body["name"], int(body["width"]), int(body["height"]), int(body.get("seed", 2)),
                           body.get("climate", "oceanic"), body.get("place", "river"), body.get("valley") or None)
        self._json({**_map_payload(body["name"]), "report": out["report"]})

    def _post_tile(self, rest, query):
        self._json({**PROJECT.save_tile(self._body()), "rev": PROJECT.tileset_rev()})

    def _post_preview(self, rest, query):
        """How an unsaved ground tile would look where it meets its neighbour in rank."""
        import render as R
        body, pal, tiles = self._body(), PROJECT.palette(), PROJECT.tiles()
        tile = {"id": body.get("id") or "preview_tile", "layer": "ground", "rank": int(body.get("rank", 0)),
                "look": R.clean_look(body.get("look"), pal, tiles)}
        frames = [R.look_sample(tile, tiles, pal, scale=2, frame=f) for f in range(R.FRAMES)]
        self._send(200, mapkit.png_bytes(R.strip(frames)), "image/png")

    def _post_motion(self, rest, query):
        """A sprite at every frame of a motion, stacked top to bottom, for the editor to play."""
        import render as R
        body, pal = self._body(), PROJECT.palette()
        art = mapkit.resolve_art(body["art"], pal)
        tile = {"variants": [art], "motion": body.get("motion") or "none"}
        P, motion = R.Palette(pal), tile["motion"]
        spr = P.sprite(art)
        # A gust is shown as it would be felt: still for most of the loop, pushed over for a moment.
        pose = (lambda f: int(f >= R.FRAMES - 6)) if motion == "gust" else (lambda f: f)
        frames = [P.image(R.moved(spr, motion, pose(f))) for f in range(R.FRAMES)]
        self._send(200, mapkit.png_bytes(R.strip(frames)), "image/png")

    def _post_art(self, rest, query):
        """Run pixelkit on tile art without saving: a recipe, a transform, or just the check."""
        body, pal = self._body(), PROJECT.palette()
        op = body.get("op")
        if op == "recipe":
            art = mapkit.recipe_art(body["recipe"], pal)
        elif op == "check":
            art = mapkit.resolve_art(body["art"], pal)
        else:
            art = mapkit.art_op(op, body["art"], body.get("params", {}), pal)
        self._json({"art": art, "problems": mapkit.pk.check(mapkit.art_image(art, pal), pal, max_colors=24)})

    def _post_export(self, rest, query):
        self._body()
        self._json(PROJECT.export())

    # -- DELETE

    def _delete_map(self, rest, query):
        self._json({"trashed": PROJECT.trash_map(rest[0])})

    def _delete_tile(self, rest, query):
        self._json(PROJECT.delete_tile(rest[0], force=query.get("force") == "1"))


def main():
    ap = argparse.ArgumentParser(description="地图设计器（网页编辑器）")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    args = ap.parse_args()
    mapkit.utf8_stdio()
    if not PROJECT.tileset()["tiles"]:
        seed.main()
    global _POOL
    from concurrent.futures import ProcessPoolExecutor
    _POOL = ProcessPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2) - 2)))
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"地图设计器: http://127.0.0.1:{args.port}/   数据: {PROJECT.data}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
