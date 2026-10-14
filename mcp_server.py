"""The map designer as an MCP server (stdio, newline-delimited JSON-RPC, no dependencies).

Registered for this project in .mcp.json. It works on the same files as the web editor, so
whatever is done here shows up in an open editor within a second or two, and the other way
round. Tools that produce pictures return them as images, so the caller can look at a tile or
a map and fix it - the same make / look / fix loop as the pixel-art skill.
"""
from __future__ import annotations

import base64
import json
import sys

import mapkit
from mapkit import MapError, Project

PROJECT = Project()
TOOLS = {}


def tool(name: str, description: str, props: dict | None = None, required: tuple = ()):
    def wrap(fn):
        TOOLS[name] = {"fn": fn, "spec": {"name": name, "description": description, "inputSchema": {
            "type": "object", "properties": props or {}, "required": list(required)}}}
        return fn
    return wrap


def image(img) -> dict:
    return {"type": "image", "data": base64.b64encode(mapkit.png_bytes(img)).decode("ascii"), "mimeType": "image/png"}


INT = {"type": "integer"}
STR = {"type": "string"}
MAP = {"type": "string", "description": "地图名"}
TILE = {"type": "string", "description": "图块 id（见 list_tiles）"}
REGION = {"type": "array", "items": INT, "minItems": 4, "maxItems": 4, "description": "[x, y, 宽, 高]，省略则是整张地图"}
ART = {"type": "object", "description": (
    "一张像素画，最大 64×64，两种写法任选其一。"
    "① 文本网格：{\"rows\": [等长的若干行], \"key\": {字符: 调色板颜色名}, \"outline\"?: 颜色名}，'.' 是透明。"
    "② 配方（pixelkit 图元）：{\"size\": [宽, 高], \"fill\"?: 颜色名, \"ops\": [...], \"outline\"?: 颜色名}，op 有 "
    "{op:'rect',x,y,w,h,color}、{op:'px',x,y,color}、"
    "{op:'blob',cx,cy,rx,ry,ramp:色阶名或[由暗到亮的颜色名]}（明暗分层的椭圆，多个叠起来就是树冠、灌木、石头）、"
    "{op:'scatter',seed,count,colors:[..],shapes:[" + "|".join(mapkit.SHAPES) + "]}、"
    "{op:'grid',rows,key,x,y}。颜色只能用 get_palette 里的名字。物件立在格子上，以画面底边中点为脚。")}
LOOK = {"type": "object", "description": (
    "地面材质的外观。地面不是逐格贴图：渲染器按材质在世界像素坐标里连续绘制，边界弯曲、纹理不重复。字段："
    "base 底色；patches [{color, scale(像素 4..96), amount(0..1), fill?(0.05..1，小于 1 打散成散点——"
    "地形内部的明暗用散点，实心色块会被看成另一种地形的边界)}] 色斑；"
    "specks [{color, shape(" + "|".join(mapkit.SHAPES) + "), density(每像素概率，约 0.002..0.02), "
    "anim?: 'twinkle'(忽隐忽现)|'drift'(滑过去，像水纹)|'gust'(只在一阵风经过时出现——草被吹倒露出的浅色)|'flow'(顺水流走)}] 细节点；"
    "edge [{color, width(像素), wobble?, density?(<1 为散点), shape?}] 内缘：靠近与低层级地形的边界时内侧怎么变；"
    "apron [同上, over?:[只洒到这些地形]] 外溢：往相邻低层级地形上洒什么（草稀到沙上、树荫投到草上、岩脚落碎石）。"
    "两种地形怎么相接是设计的重点：不要只用一条硬线，也不要只差一级颜色；"
    "rim 压在低层级材质上时的边缘色；"
    "cliff {colors:[自上而下每行一个颜色], over?:[只对这些材质], cracks?:bool} 南侧落下的崖面或土岸；"
    "shore {colors:[由近到远每像素一个颜色], lap?: bool(最近的两条来回换位，像水拍岸)} 挨着高层级材质的内侧色带；"
    "surf {foam, water, reach(像素 2..14), onto?:[地形]} 海浪：浪花涌来、冲上岸、退回、留下湿痕；"
    "flows: true 表示活水，有流向——朝它 joins 的地形（海）流，流不到哪里的（池塘）不流；细节点 anim 'flow' 顺着水流走，"
    "在不流的地形上只出现在水流到达处（河口外）；海浪在河口附近自动平息。"
    "joins [地形 id] 与这些地形连通：相接处不画任何边界（河汇入海没有岸线、没有浪）。水和水相接与水和陆地相接不是一回事；"
    "decor [{tile: 物件图块 id, density(每格概率 0..1)}] 自动散布的点缀（草丛、花、石子）。")}


def _edit(name: str, fn) -> dict:
    return PROJECT.edit(name, fn)


def _summary(name: str, what: str) -> str:
    return f"{what}。用 view_map 或 render_map 查看 {name}。"


# ---------------------------------------------------------------- maps

@tool("list_maps", "列出所有地图及其尺寸。")
def list_maps(a):
    maps = PROJECT.list_maps()
    return "\n".join(f"{m['name']}  {m['width']}×{m['height']}" for m in maps) or "还没有地图，用 create_map 新建。"


@tool("create_map", "新建一张地图。地图有两层：ground（每格一种地面材质）和 object（立在格子上的物件）。逻辑上是格子，"
                    "画出来不是：材质之间的边界是弯的，物件可以比一格大。",
      {"name": MAP, "width": INT, "height": INT,
       "fill": {"type": "string", "description": "铺满地面的 ground 图块 id，可省略"}}, ("name", "width", "height"))
def create_map(a):
    m = PROJECT.create_map(a["name"], a["width"], a["height"], a.get("fill"))
    return f"已创建 {m.name}（{m.width}×{m.height}）。"


@tool("generate_map",
      "按地理规律生成一张地图，而不是手画。先推演整个谷地：构造抬升与河流下切相抗长出地形（地貌演化模型），雨水汇成溪与河，"
      "河按流量定宽、在河漫滩上曲流、在硬岩处集中跌落；按岩性和迎浪程度生成海岸（海蚀平台、海蚀柱、卵石脊、沙嘴潟湖、盐沼）；"
      "再算温度、湿度场，按表分出生物群系，由群系决定地表、植被（草、蕨、石南、芦苇……）和树。然后切出一块放大成一格一步的地图，"
      "每格带高度、植被和群系，并保证整张走得通。给 design 则按 designs/<名>.json 的设计生成整张（place 取 whole）。"
      "返回这块地为什么长这样的说明和渲染图。",
      {"name": MAP, "width": INT, "height": INT,
       "place": {"description": "切哪里：" + "、".join(f"{k}（{v}）" for k, v in __import__("geo").PLACES.items())
                                + "；或 [x, y] 谷地格坐标（谷地 128×96）；用设计时填 whole。默认 river"},
       "design": {"type": "string", "description": "设计文件名（designs/ 下，不带 .json），按设计生成整张；此时 width/height 由设计决定"},
       "valley": {"type": "string", "description": "不填则按地貌演化生成地形；填真实谷地名则用其实测高程作对照（需先运行 dem/fetch.py）：" + "；".join(
           f"{k}＝{v['name']}（{v['about']}）" for k, v in __import__("geo").sources().items())},
       "climate": {"type": "string", "enum": list(__import__("geo").CLIMATES),
                   "description": "气候决定溪流多少、土层厚薄、雪线高低、群系分布：" + "；".join(
                       f"{k}＝{v['name']}" for k, v in __import__("geo").CLIMATES.items()) + "。默认 oceanic"},
       "seed": {"type": "integer", "description": "随机种子：不同的种子长出不同的谷地。默认 2"}},
      ("name", "width", "height"))
def generate_map(a):
    import geo
    at = a.get("place", "whole" if a.get("design") else "river")
    out = geo.make_map(PROJECT, a["name"], a["width"], a["height"], int(a.get("seed", 2)), a.get("climate", "oceanic"),
                       tuple(at) if isinstance(at, list) else at, a.get("valley") or None, a.get("design") or None)
    m = PROJECT.load_map(a["name"])
    img = mapkit.render(m, PROJECT.tiles(), PROJECT.palette(), 2)
    return [{"type": "text", "text": f"已生成 {m.name}（{m.width}×{m.height}）。\n" + "\n".join("- " + line for line in out["report"])},
            image(img)]


@tool("view_map", "用文本看地图：每格一个字符，附图例。物件会盖住它脚下的地面；指定 layer 可以只看一层。坐标原点在左上角，x 向右，y 向下。",
      {"name": MAP, "region": REGION, "layer": {"type": "string", "enum": list(mapkit.LAYERS)}}, ("name",))
def view_map(a):
    return mapkit.text_view(PROJECT.load_map(a["name"]), PROJECT.tiles(), a.get("region"), a.get("layer"))


@tool("render_map", "把地图渲染成图片返回（和游戏里看到的一样）。改完地图后用它亲眼检查；大地图用 region 分块看。",
      {"name": MAP, "region": REGION, "scale": {"type": "integer", "description": "整数放大倍数，默认 2"},
       "grid": {"type": "boolean", "description": "画格线"},
       "frame": {"type": "integer", "description": "动画循环里的第几帧（0..23，默认 0；一圈 4 秒）。画面是会动的：水纹、海浪、阵风、树、火。"
                                                    "要判断动得对不对，取两三个不同的帧对比"}}, ("name",))
def render_map(a):
    m = PROJECT.load_map(a["name"])
    img = mapkit.render(m, PROJECT.tiles(), PROJECT.palette(), a.get("scale", 2), bool(a.get("grid")), a.get("region"),
                        frame=int(a.get("frame", 0)))
    return [{"type": "text", "text": f"{m.name} {m.width}×{m.height}，图片 {img.width}×{img.height}"}, image(img)]


@tool("paint_cells", "逐格放置图块。每个图块属于固定的一层，所以只需给出坐标和图块。",
      {"name": MAP, "cells": {"type": "array", "description": "[[x, y, 图块 id], ...]",
                              "items": {"type": "array", "minItems": 3, "maxItems": 3}}}, ("name", "cells"))
def paint_cells(a):
    def fn(m, tiles):
        return mapkit.set_cells(m, tiles, [[mapkit._tile_for(tiles, t)["layer"], x, y, t] for x, y, t in a["cells"]])
    return _summary(a["name"], f"改动 {_edit(a['name'], fn)['result']} 格")


@tool("fill_rect", "用一个图块填满矩形（超出地图的部分自动裁掉）。",
      {"name": MAP, "tile": TILE, "x": INT, "y": INT, "width": INT, "height": INT},
      ("name", "tile", "x", "y", "width", "height"))
def fill_rect(a):
    r = _edit(a["name"], lambda m, t: mapkit.fill_rect(m, t, a["tile"], a["x"], a["y"], a["width"], a["height"]))
    return _summary(a["name"], f"改动 {r['result']} 格")


@tool("erase_rect", "清空矩形内某一层的格子。",
      {"name": MAP, "layer": {"type": "string", "enum": list(mapkit.LAYERS)}, "x": INT, "y": INT, "width": INT, "height": INT},
      ("name", "layer", "x", "y", "width", "height"))
def erase_rect(a):
    r = _edit(a["name"], lambda m, t: mapkit.fill_rect(m, t, None, a["x"], a["y"], a["width"], a["height"], a["layer"]))
    return _summary(a["name"], f"清空 {r['result']} 格")


@tool("draw_line", "从 (x0,y0) 到 (x1,y1) 画一条线，适合河流、道路、山脊。",
      {"name": MAP, "tile": TILE, "x0": INT, "y0": INT, "x1": INT, "y1": INT,
       "width": {"type": "integer", "description": "线宽（格），默认 1"}}, ("name", "tile", "x0", "y0", "x1", "y1"))
def draw_line(a):
    r = _edit(a["name"], lambda m, t: mapkit.draw_line(m, t, a["tile"], a["x0"], a["y0"], a["x1"], a["y1"], a.get("width", 1)))
    return _summary(a["name"], f"改动 {r['result']} 格")


@tool("flood_fill", "油漆桶：把 (x,y) 所在的、四向连通的同类区域换成指定图块。",
      {"name": MAP, "tile": TILE, "x": INT, "y": INT}, ("name", "tile", "x", "y"))
def flood_fill(a):
    r = _edit(a["name"], lambda m, t: mapkit.flood_fill(m, t, a["tile"], a["x"], a["y"]))
    return _summary(a["name"], f"改动 {r['result']} 格")


@tool("scatter_objects", "按种子随机摆放 object 图块（资源点、装饰等），只落在空格上。同样的参数得到同样的结果。",
      {"name": MAP, "tile": TILE, "count": INT, "seed": {"type": "integer", "description": "默认 1"},
       "on": {"type": "array", "items": STR, "description": "允许站在哪些 ground 图块上；省略则是任何可行走的地面"},
       "region": REGION, "spacing": {"type": "integer", "description": "与其他物件的最小间隔（格），默认 0"},
       "reachable": {"type": "boolean", "description": "只放在最大的那片可行走陆地上（资源点、出生点应当如此），默认 false"}},
      ("name", "tile", "count"))
def scatter_objects(a):
    r = _edit(a["name"], lambda m, t: mapkit.scatter(m, t, a["tile"], a["count"], a.get("seed", 1), a.get("on"),
                                                    a.get("region"), a.get("spacing", 0), bool(a.get("reachable"))))
    placed = r["result"]
    note = "" if len(placed) == a["count"] else f"（要求 {a['count']} 个，合适的空格不够）"
    return f"放了 {len(placed)} 个 {a['tile']}{note}：{placed}"


@tool("resize_map", "改变地图尺寸。原内容平移 (dx,dy) 后保留，落到新范围之外的部分丢弃。",
      {"name": MAP, "width": INT, "height": INT, "dx": INT, "dy": INT,
       "fill": {"type": "string", "description": "新露出的地面用哪个 ground 图块填，可省略"}}, ("name", "width", "height"))
def resize_map(a):
    r = _edit(a["name"], lambda m, t: mapkit.resize(m, a["width"], a["height"], a.get("dx", 0), a.get("dy", 0), a.get("fill"), t))
    return f"{a['name']} 现在是 {r['map'].width}×{r['map'].height}。"


@tool("validate_map", "结构检查：未知图块、地面空洞、从最大陆地走不到的物件算问题；陆地被水或阻挡物分成几块只作提示。并给出各图块的用量。",
      {"name": MAP}, ("name",))
def validate_map(a):
    r = mapkit.validate(PROJECT.load_map(a["name"]), PROJECT.tiles())
    head = "没有发现问题。" if r["ok"] else "发现问题：\n- " + "\n- ".join(r["problems"])
    notes = "".join(f"\n注意：{n}" for n in r["notes"])
    return head + notes + "\n统计：" + json.dumps(r["stats"], ensure_ascii=False)


# ---------------------------------------------------------------- tiles

@tool("list_tiles", "列出图块集。地面图块是材质（按层级从低到高，高的压在低的上面）；物件图块是立在格子上的精灵，可以比一格大。"
                    "每行：view_map 里的字符、id、名字、通行方式（walk 可走 / swim 需涉水 / block 阻挡）、标签。")
def list_tiles(a):
    tiles, chars = PROJECT.tiles(), mapkit.tile_chars(PROJECT.tiles())
    out = ["地面材质（层级低 → 高）："]
    for t in sorted((t for t in tiles.values() if t["layer"] == "ground"), key=lambda t: t["rank"]):
        out.append(f"  {chars[t['id']]}  {t['id']}  「{t['name']}」 层级 {t['rank']}  {t['pass']}  [{', '.join(t['tags'])}]")
    out.append("植被（长在地面上，按格记，可割、可踩、可藏身）：")
    for t in tiles.values():
        if t["layer"] == "plant":
            g = t["grow"]
            out.append(f"  {chars[t['id']]}  {t['id']}  「{t['name']}」 高 {g['height'][0]}–{g['height'][1]} 像素  {t['pass']}  [{', '.join(t['tags'])}]")
    out.append("物件：")
    for t in tiles.values():
        if t["layer"] == "object":
            sizes = "、".join(f"{len(v['rows'][0])}×{len(v['rows'])}" for v in t["variants"])
            out.append(f"  {chars[t['id']]}  {t['id']}  「{t['name']}」 {sizes}  {t['pass']}  [{', '.join(t['tags'])}]")
    return "\n".join(out) if tiles else "图块集是空的，用 save_tile 新建。"


@tool("get_tile", "取一个图块的完整定义：地面图块给出 look，物件图块给出每个变体的文本网格（rows/key）。改完用 save_tile 存回去。",
      {"id": TILE}, ("id",))
def get_tile(a):
    t = mapkit._tile_for(PROJECT.tiles(), a["id"])
    if t["layer"] in ("ground", "plant"):
        return json.dumps(t, ensure_ascii=False, indent=1)
    head = json.dumps({k: v for k, v in t.items() if k != "variants"}, ensure_ascii=False)
    arts = [f"变体 {i}  key: {json.dumps(v['key'], ensure_ascii=False)}\n" + "\n".join(v["rows"])
            for i, v in enumerate(t["variants"])]
    return head + "\n" + "\n".join(arts)


@tool("save_tile",
      "新建图块，或修改同 id 的图块（只给要改的字段即可）。layer=ground 是地面材质，给 look；layer=plant 是植被（草），给 grow，"
      "它不是贴图，而是按格长满、边缘越出格子的一层；layer=object 是物件，给 art（一张）"
      "或 variants（多张，地图上每格按位置挑一张）。返回预览图和 pixelkit 的检查结果，请看图确认：材质与相邻材质相接是否自然；"
      "物件在 1× 下是否认得出、光从左上来、有 1 像素边距并描边（树冠用 g0、其他用 n0）。",
      {"id": {"type": "string", "description": "小写字母开头，只含小写字母、数字、下划线"},
       "name": {"type": "string", "description": "显示名"},
       "layer": {"type": "string", "enum": list(mapkit.LAYERS)},
       "pass": {"type": "string", "enum": list(mapkit.PASS)},
       "tags": {"type": "array", "items": STR, "description": "自由标签，游戏据此赋予图块含义（如 resource、wood、marker、decor）"},
       "rank": {"type": "integer", "description": "地面：层级，高的压在低的上面（水最低，山最高）。省略则排到最高"},
       "look": LOOK,
       "grow": {"type": "object", "description": (
           "植被怎么长：height [最矮, 最高] 草叶高几像素（1..16，长草 6..12、短草 2..4）；density 每像素有一根草叶的机会（0.01..0.8）；"
           "colors 从根到尖 2..6 个颜色；under 草丛里露出的地色；spill 越出本格多少像素（长草的叶子会伸到上一格）。")},
       "art": ART,
       "variants": {"type": "array", "items": ART, "description": "物件：1 到 8 张像素画"},
       "shadow": {"type": "boolean", "description": "物件：脚下投一块阴影"},
       "flip": {"type": "boolean", "description": "物件：每格随机左右镜像"},
       "jitter": {"type": "integer", "description": "物件：每格位置的随机偏移，0..6 像素；让成片的树不排成行列"},
       "motion": {"type": "string", "enum": ["none", "sway", "flicker", "bob", "gust"],
                  "description": "物件：怎么动。动画由这一张图推出来，不用逐帧画：sway 上半部左右摆（树、草）、"
                                 "flicker 顶部抖动（火）、bob 身体上下一像素（人、兽）、gust 平时不动、一阵风经过时倒伏（草丛、花）。"
                                 "每个摆在图上的物件相位不同"},
       "footprint": {"type": "array", "items": INT, "minItems": 2, "maxItems": 2,
                     "description": "物件：占地 [宽, 深]（格）。物件立在前沿正中那一格，向两侧和后方占地；"
                                    "pass=block 时这些格子都走不过去。比例尺约一格一步，房子应占 3×2 左右"}}, ("id",))
def save_tile(a):
    keys = ("id", "name", "layer", "pass", "tags", "rank", "look", "grow", "art", "variants", "shadow", "flip", "jitter", "footprint", "motion")
    r = PROJECT.save_tile({k: a.get(k) for k in keys})
    t = r["tile"]
    check = "检查通过。" if not r["problems"] else "检查发现：\n- " + "\n- ".join(r["problems"])
    return [{"type": "text", "text": f"已保存 {t['id']}「{t['name']}」（{t['layer']}，{t['pass']}）。{check}"},
            image(mapkit.tile_preview(t, PROJECT.palette(), PROJECT.tiles()))]


@tool("preview_tile", "返回图块的预览图。地面材质：它与相邻层级材质相接的一小块；物件：每个变体放大后在深浅两种底色上，上方是 1× 原大。",
      {"id": TILE}, ("id",))
def preview_tile(a):
    tiles = PROJECT.tiles()
    t = mapkit._tile_for(tiles, a["id"])
    return [{"type": "text", "text": f"{t['id']}「{t['name']}」"}, image(mapkit.tile_preview(t, PROJECT.palette(), tiles))]


@tool("transform_tile", "对物件图块的一张像素画做一次 pixelkit 变换并保存：outline 描边、mirror 左右镜像、shift 平移（回绕）、swap 换色。",
      {"id": TILE, "op": {"type": "string", "enum": ["outline", "mirror", "shift", "swap"]},
       "variant": {"type": "integer", "description": "第几个变体，默认 0"},
       "color": {"type": "string", "description": "outline 用的颜色名，默认 n0"},
       "dx": INT, "dy": INT,
       "map": {"type": "object", "description": "swap 用：{原颜色名: 新颜色名}"},
       "as_variant": {"type": "boolean", "description": "不改原图，把结果追加为这个图块的新变体"},
       "save_as": {"type": "string", "description": "另存为这个新 id，不动原图块"}}, ("id", "op"))
def transform_tile(a):
    src = mapkit._tile_for(PROJECT.tiles(), a["id"])
    if src["layer"] != "object":
        raise MapError("transform_tile 只用于物件图块；地面材质请改 look")
    i = int(a.get("variant", 0))
    if not 0 <= i < len(src["variants"]):
        raise MapError(f"{a['id']} 只有 {len(src['variants'])} 个变体")
    art = mapkit.art_op(a["op"], src["variants"][i], a, PROJECT.palette())
    arts = list(src["variants"])
    if a.get("as_variant"):
        arts.append(art)
    else:
        arts[i] = art
    keep = {k: src[k] for k in ("name", "layer", "pass", "tags", "shadow", "flip", "jitter", "footprint", "motion")}
    return save_tile({**keep, "id": a.get("save_as") or a["id"], "variants": arts})


@tool("delete_tile", "删除图块。还有地图在用、或它是某个材质的点缀时会拒绝；force 为真则连同这些用处一起清掉。",
      {"id": TILE, "force": {"type": "boolean"}}, ("id",))
def delete_tile(a):
    r = PROJECT.delete_tile(a["id"], bool(a.get("force")))
    return f"已删除 {r['deleted']}。" + (f"清掉的格子：{r['cleared']}" if r["cleared"] else "")


@tool("get_palette", "项目调色板：所有颜色名及色值，以及色阶（由暗到亮，暗部偏冷、亮部偏暖）。图块只能用这里的颜色。")
def get_palette(a):
    p = PROJECT.palette_json()
    colors = "  ".join(f"{k}={v}" for k, v in p["colors"].items())
    ramps = "\n".join(f"  {k}: {' → '.join(v)}" for k, v in p.get("ramps", {}).items())
    return (f"颜色：{colors}\n色阶：\n{ramps}\n（body/acc 等是给角色换色用的占位色，图块不要用。"
            "同一字母的颜色数字越小越暗；阴影就是把地面颜色换成暗一级的。）")


@tool("export_tileset", "导出给游戏（都是可动的数据，不是一张死图）：物件精灵逐帧打包成 export/tiles.png + tiles.json；"
                        "每张地图在 export/maps/ 下有静止的地面、各帧相对它的差异层，以及所有物件与点缀的位置、变体、相位清单。")
def export_tileset(a):
    r = PROJECT.export()
    maps = "；".join(f"{m['name']}（会动的帧 {m['moving_frames']}，物件与点缀 {m['things']} 个）" for m in r["maps"]) or "无"
    return (f"导出 {r['objects']} 个物件的逐帧图集 → {r['png']}、{r['json']}；地图 {len(r['maps'])} 张：{maps}。"
            + ("" if not r["problems"] else "检查：" + "；".join(r["problems"])))


# ---------------------------------------------------------------- protocol

def _call(name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"content": [{"type": "text", "text": f"没有工具 {name}"}], "isError": True}
    try:
        out = TOOLS[name]["fn"](args or {})
    except MapError as e:
        return {"content": [{"type": "text", "text": str(e)}], "isError": True}
    except (KeyError, TypeError, ValueError) as e:
        return {"content": [{"type": "text", "text": f"参数有误：{e!r}"}], "isError": True}
    except Exception as e:  # a failed call must not take the server down with it
        return {"content": [{"type": "text", "text": f"出错了：{e!r}"}], "isError": True}
    return {"content": out if isinstance(out, list) else [{"type": "text", "text": out}]}


def handle(msg: dict):
    """One JSON-RPC message in, the reply out (None for notifications)."""
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2024-11-05"),
                  "capabilities": {"tools": {}},
                  "serverInfo": {"name": "mapdesigner", "version": "0.1.0"},
                  "instructions": "像素地图设计器。地图逻辑上是格子，画面是星露谷式的连续地表：地面图块是材质，"
                                  "物件图块是可跨多格的精灵。先 list_tiles / list_maps，再编辑；改完用 render_map 看图、"
                                  "用 validate_map 做结构检查。新图块用 save_tile，颜色只能取自 get_palette。"
                                  "想让地形自然：材质成片而不是横平竖直的条块，同类物件用 scatter_objects 成群摆放。"
                                  "比例尺接近 1:1：一格约一步，人是 16 像素高；画的是人走得到的一处地方（空地、池塘、小溪、"
                                  "小路、岩台），不是把山海缩进一屏的大地图。图块集里的 person 可以摆在图上当比例参照。"}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": [t["spec"] for t in TOOLS.values()]}
    elif method == "tools/call":
        params = msg.get("params", {})
        result = _call(params.get("name"), params.get("arguments"))
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main():
    out = sys.stdout.buffer
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        try:
            reply = handle(json.loads(line.decode("utf-8")))
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        if reply is not None:
            out.write(json.dumps(reply).encode("ascii") + b"\n")
            out.flush()


if __name__ == "__main__":
    main()
