"""SVG + DXF + JSON для всіх схем.  Запуск: python3 build_vectors.py"""
import json
import math
import os

import ezdxf
from ezdxf import colors
from shapely.geometry import Polygon, box

from schemes import SCHEMES, PRODUCT, build, corners, stats, validate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AREA = 2400  # мм, сторона квадратної ділянки для креслення

FILL = {"H": "#b9b3ad", "V": "#5d5856"}  # поздовжні світлі, поперечні темні
RGB = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) for k, v in FILL.items()}
JOINT = "#26282b"


def tiled(px, py, stones, x0, y0, x1, y1):
    """Усі камені, що потрапляють у прямокутник [x0,x1]x[y0,y1]."""
    out = []
    reach = 400
    i0, i1 = math.floor((x0 - reach) / px), math.ceil((x1 + reach) / px)
    j0, j1 = math.floor((y0 - reach) / py), math.ceil((y1 + reach) / py)
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            for cx, cy, w, h, a, k in stones:
                x, y = cx + i * px, cy + j * py
                if x0 - reach < x < x1 + reach and y0 - reach < y < y1 + reach:
                    out.append((x, y, w, h, a, k))
    return out


def svg(s, px, py, stones):
    items = tiled(px, py, stones, 0, 0, AREA, AREA)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-40 -40 {AREA + 80} {AREA + 160}" '
             f'width="{(AREA + 80) // 4}" height="{(AREA + 160) // 4}">',
             f'<title>{s["name"]} — схема викладки</title>',
             '<defs><clipPath id="c"><rect width="%d" height="%d"/></clipPath></defs>' % (AREA, AREA),
             f'<rect x="-40" y="-40" width="{AREA + 80}" height="{AREA + 160}" fill="#ffffff"/>',
             f'<g clip-path="url(#c)" stroke="{JOINT}" stroke-width="5" stroke-linejoin="round">']
    for st in items:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in corners(st))
        parts.append(f'<polygon points="{pts}" fill="{FILL[st[5]]}"/>')
    parts.append("</g>")
    parts.append(f'<rect width="{AREA}" height="{AREA}" fill="none" stroke="{JOINT}" stroke-width="10"/>')
    parts.append(f'<text x="0" y="{AREA + 90}" font-family="Arial, sans-serif" font-size="64" fill="#222">'
                 f'{s["name"]} · ділянка {AREA / 1000:.1f}×{AREA / 1000:.1f} м · модуль {px:.0f}×{py:.0f} мм</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def dxf(s, px, py, stones):
    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.MM
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    layer_color = {"H": 253, "V": 250}
    for k, c in layer_color.items():
        doc.layers.add(f"PLYTA_{k}", color=c)
        doc.layers.add(f"PLYTA_{k}_FILL", color=c)
    doc.layers.add("PLYTA_CUT", color=1)
    doc.layers.add("BOUNDARY", color=7)
    doc.layers.add("MODULE", color=3)
    doc.layers.add("TEXT", color=7)

    def add_stone(target, pts, k, layer=None, hatch=True):
        layer = layer or f"PLYTA_{k}"
        target.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer})
        if hatch:
            h = target.add_hatch(dxfattribs={"layer": f"PLYTA_{k}_FILL"})
            h.rgb = RGB[k]
            h.paths.add_polyline_path(pts, is_closed=True)

    # Блок модуля (один період) — для масиву/копіювання
    blk = doc.blocks.new(name=f"MODULE_{s['id']}")
    for st in stones:
        add_stone(blk, corners(st), st[5])
    blk.add_lwpolyline([(0, 0), (px, 0), (px, py), (0, py)], close=True, dxfattribs={"layer": "MODULE"})

    # Ділянка AREA x AREA з підрізкою країв. Y у DXF дивиться вгору — віддзеркалюємо.
    flip = lambda pts: [(x, AREA - y) for x, y in pts]
    area = box(0, 0, AREA, AREA)
    whole = cut = 0
    for st in tiled(px, py, stones, 0, 0, AREA, AREA):
        poly = Polygon(corners(st))
        inter = poly.intersection(area)
        if inter.is_empty or inter.area < 1:
            continue
        if inter.area > poly.area - 1:
            add_stone(msp, flip(list(poly.exterior.coords)[:-1]), st[5])
            whole += 1
        else:
            geoms = getattr(inter, "geoms", [inter])
            for g in geoms:
                if g.geom_type == "Polygon":
                    add_stone(msp, flip(list(g.exterior.coords)[:-1]), st[5], layer="PLYTA_CUT")
            cut += 1
    msp.add_lwpolyline([(0, 0), (AREA, 0), (AREA, AREA), (0, AREA)], close=True, dxfattribs={"layer": "BOUNDARY"})
    msp.add_text(f"{s['name']} - ділянка {AREA}x{AREA} мм; цілих {whole}, підрізаних {cut}",
                 height=60, dxfattribs={"layer": "TEXT"}).set_placement((0, -120))
    ox = AREA + 400
    msp.add_blockref(f"MODULE_{s['id']}", (ox, AREA - py), dxfattribs={"yscale": 1})
    msp.add_text(f"Модуль (період) {px:.1f}x{py:.1f} мм - блок MODULE_{s['id']}",
                 height=50, dxfattribs={"layer": "TEXT"}).set_placement((ox, AREA - py - 100))
    return doc, whole, cut


def main():
    os.makedirs(f"{ROOT}/svg", exist_ok=True)
    os.makedirs(f"{ROOT}/dxf", exist_ok=True)
    data = {"product": PRODUCT, "area_mm": AREA, "schemes": []}
    for s in SCHEMES:
        px, py, stones = build(s)
        lo, hi, fill = validate(px, py, stones)
        assert lo == hi == 1 and abs(fill - 1) < 1e-6, s["id"]
        with open(f"{ROOT}/svg/{s['id']}.svg", "w") as f:
            f.write(svg(s, px, py, stones))
        doc, whole, cut = dxf(s, px, py, stones)
        doc.saveas(f"{ROOT}/dxf/{s['id']}.dxf")
        data["schemes"].append({
            "id": s["id"], "name": s["name"], "note": s["note"], "use": s["use"],
            "period_mm": [round(px, 3), round(py, 3)],
            "stones": [[round(cx, 3), round(cy, 3), w, h, a, k] for cx, cy, w, h, a, k in stones],
            "stats": stats(px, py, stones),
            "area_sample": {"whole": whole, "cut": cut},
        })
        print(s["id"], "ok", whole, cut)
    with open(f"{ROOT}/schemes.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
