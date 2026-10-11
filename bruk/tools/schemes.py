"""Схеми викладки бруківки 90x120 / 120x120 / 180x120 мм.

Кожна схема — періодична плитка: прямокутник періоду (px, py) у мм і
список каменів, центри яких лежать у цьому прямокутнику. Повторюючи
період по осях X та Y, отримуємо безшовне мощення будь-якого розміру.

Камінь: (cx, cy, w, h, angle_deg, size) — центр, розміри вздовж осей
каменя, кут повороту, ключ формату ("90x120" | "120x120" | "180x120").
Вісь Y спрямована вниз (як у SVG і в зображеннях текстур).
"""
import math
import random

ROW = 120  # ширина ряду: усі три формати мають сторону 120 мм

SIZES = {
    "90x120": (90, 120),
    "120x120": (120, 120),
    "180x120": (180, 120),
}

PRODUCT = {
    "name": "Бруківка «Старе місто», сіра",
    "source": "unibruk.com.ua",
    "color": "Сіра",
    "sizes_mm": ["90x120", "120x120", "180x120"],
    "variants": [
        {"height_mm": 45, "m2_per_pallet": 16.85, "kg_per_m2": 102.3, "price_uah_m2": 630.00},
        {"height_mm": 60, "m2_per_pallet": 14.04, "kg_per_m2": 136.8, "price_uah_m2": 695.00},
    ],
}


def _key(length, across=ROW):
    a, b = sorted((length, across))
    return f"{a}x{b}" if (a, b) != (120, 180) else "180x120"


def size_key(w, h):
    a, b = sorted((round(w), round(h)))
    return {(90, 120): "90x120", (120, 120): "120x120", (120, 180): "180x120"}[(a, b)]


def rows(px, spec):
    """spec: список (зсув, [довжини]) — горизонтальні ряди висотою 120."""
    stones = []
    for r, (offset, lengths) in enumerate(spec):
        assert sum(lengths) == px, (px, lengths)
        x = offset
        for L in lengths:
            cx = (x + L / 2) % px
            stones.append((cx, r * ROW + ROW / 2, L, ROW, 0, size_key(L, ROW)))
            x += L
    return px, ROW * len(spec), stones


def joints(offset, lengths, px):
    out, x = [], offset
    for L in lengths:
        out.append(x % px)
        x += L
    return out


def circ_dist(a, b, p):
    d = abs(a - b) % p
    return min(d, p - d)


# ---------------------------------------------------------------- схеми

def old_town_random(seed=7):
    px, n_rows = 1080, 9  # 1080x1080 — квадратний модуль
    rng = random.Random(seed)

    def make_row():
        while True:
            seq, s = [], 0
            while s < px:
                L = rng.choice((90, 120, 180))
                seq.append(L)
                s += L
            if s == px:
                return seq

    def ok(a, b):
        ja, jb = joints(*a, px), joints(*b, px)
        return min(circ_dist(x, y, px) for x in ja for y in jb) >= 30

    while True:
        spec = []
        for r in range(n_rows):
            for _ in range(500):
                cand = (rng.randrange(0, px, 15), make_row())
                if not spec or ok(spec[-1], cand):
                    break
            spec.append(cand)
        if ok(spec[-1], spec[0]):
            return rows(px, spec)


def rhythm_rows():
    return rows(480, [(0, [180, 120, 90, 90]), (255, [180, 90, 90, 120])])


def cross_bands():
    """Смуга вертикальних 180x120 (впоперек) + ряд квадратів зі зсувом на півкаменю."""
    stones = [(60, 90, 120, 180, 0, "180x120"),       # смуга висотою 180
              (0, 180 + 60, 120, 120, 0, "120x120")]   # ряд квадратів, зсув 60
    return 120, 300, stones


def bands():
    return rows(360, [(0, [180, 180]), (30, [120, 120, 120]),
                      (60, [180, 180]), (30, [90, 90, 90, 90])])


def _herringbone(L=180, W=120):
    """Ялинка 90°: пари (горизонтальний + вертикальний) вздовж діагоналі."""
    px = py = 720
    cells = []
    for a in range(-12, 13):
        for b in range(-12, 13):
            ox, oy = a * W + b * L, a * W - b * L
            # горизонтальний [0,L]x[0,W]; вертикальний [L,L+W]x[W-L,W]
            cells.append((ox + L / 2, oy + W / 2, L, W))
            cells.append((ox + L + W / 2, oy + W - L / 2, W, L))
    stones = [(cx % px, cy % py, w, h, 0, "180x120")
              for cx, cy, w, h in cells if 0 <= cx < px and 0 <= cy < py]
    return px, py, stones


def herringbone90():
    return _herringbone()


def herringbone45():
    L, W = 180, 120
    px, py = L * math.sqrt(2), W * math.sqrt(2)
    c = s = math.sqrt(0.5)  # поворот на +45°
    stones = []
    for a in range(-6, 7):
        for b in range(-6, 7):
            ox, oy = a * W + b * L, a * W - b * L
            for cx, cy, w, h in ((ox + L / 2, oy + W / 2, L, W),
                                 (ox + L + W / 2, oy + W - L / 2, W, L)):
                rx, ry = c * cx - s * cy, s * cx + c * cy
                stones.append((rx, ry, w, h, 45))
    out, seen = [], set()
    for rx, ry, w, h, ang in stones:
        x, y = rx % px, ry % py
        k = (round(x, 3) % round(px, 3), round(y, 3) % round(py, 3), w)
        if k in seen:
            continue
        seen.add(k)
        out.append((x, y, w, h, ang, "180x120"))
    return px, py, out


def basketweave():
    """Плетінка: блоки 360x360, горизонтальні й вертикальні в шаховому порядку."""
    pattern = [[180, 180], [90, 180, 90], [180, 180]]
    stones = []
    for bi in range(2):
        for bj in range(2):
            ox, oy = bi * 360, bj * 360
            horiz = (bi + bj) % 2 == 0
            for r, lens in enumerate(pattern):
                t = 0
                for L in lens:
                    if horiz:
                        stones.append((ox + t + L / 2, oy + r * ROW + ROW / 2, L, ROW, 0, size_key(L, ROW)))
                    else:
                        stones.append((ox + r * ROW + ROW / 2, oy + t + L / 2, ROW, L, 0, size_key(L, ROW)))
                    t += L
    return 720, 720, stones


def _pinwheel(L, W, arm_rows, center, period):
    """Вітряк: лопаті LxW навколо квадрата (L-W). arm_rows — ряди лопаті
    від зовнішнього краю; center — функція, що заповнює квадрат."""
    c = L - W

    def arm_h(x0, y0):
        out = []
        for r, lens in enumerate(arm_rows):
            t = 0
            for Ln in lens:
                out.append((x0 + t + Ln / 2, y0 + r * ROW + ROW / 2, Ln, ROW))
                t += Ln
        return out

    def arm_v(x0, y0):  # поворот лопаті на 90°: зовнішній ряд — праворуч
        out = []
        for r, lens in enumerate(reversed(arm_rows)):
            t = 0
            for Ln in lens:
                out.append((x0 + r * ROW + ROW / 2, y0 + t + Ln / 2, ROW, Ln))
                t += Ln
        return out

    cell = center(c) + arm_h(0, -W) + arm_v(c, 0)
    px = py = period
    stones = []
    for a in range(-10, 11):
        for b in range(-10, 11):
            ox, oy = a * L - b * W, a * W + b * L
            for cx, cy, w, h in cell:
                x, y = cx + ox, cy + oy
                if 0 <= x < px and 0 <= y < py:
                    stones.append((x, y, w, h, 0, size_key(w, h)))
    return px, py, stones


def pinwheel():
    """Вітряк: чотири «лопаті» 360x240 навколо квадрата 120x120."""
    return _pinwheel(360, 240, ([180, 180], [90, 180, 90]),
                     lambda c: [(c / 2, c / 2, c, c)], 1560)


def framed_square():
    """Квадрат у рамці: 2x2 квадрати 120 в обрамленні пар 180x120, що обертаються."""
    return _pinwheel(360, 120, ([180, 180],),
                     lambda c: [(60 + i * 120, 60 + j * 120, 120, 120) for i in range(2) for j in range(2)], 1200)


def checker():
    """Шахова мозаїка: блоки 360x360 — 3x3 квадрати та паркет із 90x120."""
    stones = []
    for bi in range(2):
        for bj in range(2):
            ox, oy = bi * 360, bj * 360
            if (bi + bj) % 2 == 0:
                for i in range(3):
                    for j in range(3):
                        stones.append((ox + i * 120 + 60, oy + j * 120 + 60, 120, 120, 0, "120x120"))
            else:
                for i in range(4):
                    for j in range(3):
                        stones.append((ox + i * 90 + 45, oy + j * 120 + 60, 90, 120, 0, "90x120"))
    return 720, 720, stones


SCHEMES = [
    dict(id="01_old_town", name="Старе місто", fn=old_town_random,
         note="Вільні ряди з усіх трьох форматів, шви сусідніх рядів не збігаються. Класична «бруківка старого міста».",
         use="Тротуари, двори, площі"),
    dict(id="02_rhythm", name="Ритмічні ряди", fn=rhythm_rows,
         note="Набір 180·120·90·90 у кожному ряду, у сусідніх рядах інший порядок; шви розходяться щонайменше на 45 мм.",
         use="Доріжки, під'їзди"),
    dict(id="03_cross_bands", name="Поперечні смуги", fn=cross_bands,
         note="Смуга з 180×120, покладених упоперек, чергується з рядом квадратів 120×120 зі зсувом на півкаменю.",
         use="Доріжки, тераси"),
    dict(id="04_bands", name="Смуги трьох форматів", fn=bands,
         note="Ряди 180 → 120 → 180 → 90. Чітка поперечна смуга, добре зорово «розширює» вузьку доріжку.",
         use="Вузькі доріжки, тротуари"),
    dict(id="05_framed_square", name="Квадрат у рамці", fn=framed_square,
         note="Група 2×2 квадрати 120×120 в обрамленні пар 180×120, що обертаються навколо неї.",
         use="Двори, майданчики"),
    dict(id="06_herringbone90", name="Ялинка 90°", fn=herringbone90,
         note="Лише 180×120. Найміцніше зчеплення каменів, рекомендовано для заїзду авто.",
         use="Під'їзди, паркування (60 мм)"),
    dict(id="07_herringbone45", name="Ялинка 45°", fn=herringbone45,
         note="Лише 180×120, ялинка під кутом 45° до бордюру. Потребує підрізки біля країв.",
         use="Під'їзди, паркування (60 мм)"),
    dict(id="08_basketweave", name="Плетінка", fn=basketweave,
         note="Квадратні блоки 360×360 з 180 та 90 у шаховому порядку: горизонтальні та вертикальні.",
         use="Тераси, зони відпочинку"),
    dict(id="09_pinwheel", name="Вітряк", fn=pinwheel,
         note="Чотири лопаті 360×240 обертаються навколо квадрата 120×120. Використовує всі три формати.",
         use="Площі, акцентні зони"),
    dict(id="10_checker", name="Шахова мозаїка", fn=checker,
         note="Блоки 3×3 квадрати 120×120 чергуються з паркетом із 90×120.",
         use="Тераси, вхідні групи"),
]


def build(s):
    px, py, stones = s["fn"]()
    return px, py, stones


def corners(st):
    cx, cy, w, h, ang, _ = st
    a = math.radians(ang)
    ca, sa = math.cos(a), math.sin(a)
    pts = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
    return pts


def validate(px, py, stones, step=None):
    """Перевіряє, що період покритий рівно один раз (без щілин і накладань)."""
    import numpy as np
    step = step or 5.0
    xs = (np.arange(0, px, step) + step * 0.371)
    ys = (np.arange(0, py, step) + step * 0.613)
    X, Y = np.meshgrid(xs, ys)
    cnt = np.zeros_like(X, dtype=int)
    for cx, cy, w, h, ang, _ in stones:
        a = math.radians(ang)
        ca, sa = math.cos(a), math.sin(a)
        for sx in (-px, 0, px):
            for sy in (-py, 0, py):
                dx, dy = X - (cx + sx), Y - (cy + sy)
                u = dx * ca + dy * sa
                v = -dx * sa + dy * ca
                cnt += (np.abs(u) < w / 2) & (np.abs(v) < h / 2)
    area = sum(w * h for _, _, w, h, _, _ in stones)
    return int(cnt.min()), int(cnt.max()), area / (px * py)


def stats(px, py, stones):
    per = {k: 0 for k in SIZES}
    for st in stones:
        per[st[5]] += 1
    m2 = px * py / 1e6
    out = {}
    for k, n in per.items():
        w, h = SIZES[k]
        out[k] = {"pcs_per_m2": round(n / m2, 2), "area_share": round(n * w * h / (px * py), 4)}
    return out


if __name__ == "__main__":
    for s in SCHEMES:
        px, py, st = build(s)
        lo, hi, fill = validate(px, py, st)
        print(f"{s['id']:22s} period {px:7.1f} x {py:7.1f}  stones {len(st):3d}  cover {lo}..{hi}  fill {fill:.4f}",
              {k: v['area_share'] for k, v in stats(px, py, st).items()})
