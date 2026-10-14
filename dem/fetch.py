"""Fetch the real elevation windows in this folder again (or add one).

    python tools/mapdesigner/dem/fetch.py              # every valley in sources.json
    python tools/mapdesigner/dem/fetch.py glen_brittle  # just one

Source: Terrain Tiles (Mapzen "terrarium" encoding) on AWS Open Data,
https://registry.opendata.aws/terrain-tiles/ - PNG tiles where the height in metres is
R*256 + G + B/256 - 32768. Nine tiles at zoom 11 around each valley's point, cropped to
192x144 cells with the point low in the window (the valley runs up from its mouth).

To add a valley: put its name, lat, lon, climate and a line about it in sources.json, then
run this for it. It fills in cell_m. See docs/引用数据.md for what is owed to the source.
"""
import io
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ZOOM, W, H = 11, 192, 144


def tile(z, x, y):
    req = urllib.request.Request(URL.format(z=z, x=x, y=y), headers={"User-Agent": "mapdesigner-terrain-study/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        rgb = np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"), dtype=np.float64)
    return rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768


def main():
    path = HERE / "sources.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for name in sys.argv[1:] or list(data["valleys"]):
        v = data["valleys"][name]
        n = 2 ** ZOOM
        fx = (v["lon"] + 180) / 360 * n
        fy = (1 - math.asinh(math.tan(math.radians(v["lat"]))) / math.pi) / 2 * n
        tx, ty = int(fx), int(fy)
        big = np.block([[tile(ZOOM, tx + dx, ty + dy) for dx in (-1, 0, 1)] for dy in (-1, 0, 1)])
        px, py = int((fx - tx + 1) * 256), int((fy - ty + 1) * 256)
        top, left = max(py - H + 28, 0), max(px - W // 2, 0)
        np.save(HERE / f"{name}.npy", big[top:top + H, left:left + W].astype(np.float32))
        v["cell_m"] = round(156543.03 * math.cos(math.radians(v["lat"])) / n, 1)
        print(f"{name}: {v['cell_m']} m a cell")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
