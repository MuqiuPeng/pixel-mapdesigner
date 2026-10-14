"""Drive mcp_server.py over stdio the way a client would, against a throwaway data folder.

    python tools/mapdesigner/mcp_smoke.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "MAPDESIGNER_DATA": str(Path(tmp) / "maps"), "MAPDESIGNER_EXPORT": str(Path(tmp) / "out")}
        subprocess.run([sys.executable, str(HERE / "seed.py")], env=env, check=True, stdout=subprocess.DEVNULL)
        proc = subprocess.Popen([sys.executable, str(HERE / "mcp_server.py")], env=env,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        n = 0

        def rpc(method, params=None, notify=False):
            nonlocal n
            msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
            if not notify:
                n += 1
                msg["id"] = n
            proc.stdin.write(json.dumps(msg, ensure_ascii=False).encode("utf-8") + b"\n")
            proc.stdin.flush()
            if notify:
                return None
            reply = json.loads(proc.stdout.readline())
            assert reply["id"] == n, reply
            return reply

        def call(tool_name, **args):
            r = rpc("tools/call", {"name": tool_name, "arguments": args})["result"]
            return r.get("isError", False), r["content"]

        init = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "smoke"}})
        assert init["result"]["serverInfo"]["name"] == "mapdesigner", init
        assert init["result"]["protocolVersion"] == "2025-06-18"
        rpc("notifications/initialized", notify=True)
        tools = rpc("tools/list")["result"]["tools"]
        print(len(tools), "tools:", ", ".join(t["name"] for t in tools))
        assert all(t["inputSchema"]["type"] == "object" for t in tools)

        err, out = call("create_map", name="岛", width=12, height=8, fill="water")
        assert not err, out
        assert not call("fill_rect", name="岛", tile="grass", x=2, y=1, width=8, height=6)[0]
        assert not call("draw_line", name="岛", tile="forest", x0=3, y0=2, x1=8, y1=5, width=2)[0]
        # A new material (by its look) and a new object (by a pixelkit recipe, taller than a cell).
        err, out = call("save_tile", id="snow", name="雪地", layer="ground", look={
            "base": "f0", "patches": [{"color": "s4", "scale": 20, "amount": 0.3}], "rim": "s3",
            "specks": [{"color": "b3", "shape": "dot", "density": 0.01}]})
        assert not err and out[1]["type"] == "image" and "检查通过" in out[0]["text"], out[0]
        err, out = call("save_tile", id="totem", name="图腾", layer="object", shadow=True, art={
            "size": [10, 30], "ops": [{"op": "rect", "x": 3, "y": 2, "w": 4, "h": 27, "color": "w2"},
                                      {"op": "blob", "cx": 5, "cy": 6, "rx": 3.5, "ry": 3.5, "ramp": "fire"}],
            "outline": "n0"})
        assert not err and out[1]["type"] == "image" and "检查通过" in out[0]["text"], out[0]
        assert "10×30" in call("list_tiles")[1][0]["text"]
        assert not call("flood_fill", name="岛", tile="snow", x=2, y=1)[0]
        assert not call("paint_cells", name="岛", cells=[[6, 3, "totem"]])[0]
        err, out = call("scatter_objects", name="岛", tile="tree", count=4, on=["forest"], seed=3)
        assert not err and "放了 4 个" in out[0]["text"], out
        assert not call("paint_cells", name="岛", cells=[[2, 1, "spawn"], [0, 0, "rock"]])[0]
        err, out = call("view_map", name="岛")
        print(out[0]["text"])
        err, out = call("validate_map", name="岛")
        assert "rock@(0,0)" in out[0]["text"], out[0]["text"]
        err, out = call("render_map", name="岛", scale=2)
        assert out[1]["mimeType"] == "image/png" and len(out[1]["data"]) > 500
        assert not call("transform_tile", id="tree", op="swap", map={"g2": "y1", "g3": "y2"}, as_variant=True)[0]
        assert "变体 3" in call("get_tile", id="tree")[1][0]["text"]
        assert '"base": "f0"' in call("get_tile", id="snow")[1][0]["text"]
        assert call("preview_tile", id="water")[1][1]["type"] == "image"
        assert "leaf: g0" in call("get_palette")[1][0]["text"]
        err, out = call("export_tileset")
        assert not err and "地图 3 张" in out[0]["text"] and "会动的帧" in out[0]["text"], out
        still, later = (call("render_map", name="岛", scale=1, frame=f)[1][1]["data"] for f in (0, 3))
        assert still != later, "the picture should move between frames"
        assert not call("save_tile", id="totem", motion="sway")[0]
        assert call("save_tile", id="totem", motion="dance")[0]

        # Failures come back as tool errors, and the server keeps going.
        for tool_name, args in [("fill_rect", dict(name="岛", tile="lava", x=0, y=0, width=1, height=1)),
                           ("view_map", dict(name="nowhere")), ("fill_rect", dict(name="岛")),
                           ("delete_tile", dict(id="snow")), ("delete_tile", dict(id="tuft")),
                           ("save_tile", dict(id="mud", layer="ground", look={"base": "nope"})),
                           ("transform_tile", dict(id="water", op="mirror")), ("no_such_tool", {})]:
            err, out = call(tool_name, **args)
            assert err, (tool_name, out)
            print("refused:", tool_name, "->", out[0]["text"][:60])
        assert rpc("nope/nope")["error"]["code"] == -32601
        assert rpc("ping")["result"] == {}
        proc.stdin.close()
        assert proc.wait(5) == 0
    print("MCP smoke test passed")


if __name__ == "__main__":
    main()
