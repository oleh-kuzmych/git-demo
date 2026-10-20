"""Схеми викладки тротуарної плити 400x200x60 мм.

Кожна схема — періодична плитка: прямокутник періоду (px, py) у мм і
список плит, центри яких лежать у цьому прямокутнику.

Плита: (cx, cy, w, h, angle_deg, key) — центр, розміри вздовж осей плити,
кут повороту, орієнтація "H" (довга сторона вздовж X) або "V".
Вісь Y спрямована вниз (як у SVG і в зображеннях текстур).
"""
import math

L, W = 400, 200

PRODUCT = {
    "name": "Плита тротуарна 400×200×60",
    "source": "технічне завдання на форму ВР-2664 (03.2021) + фото викладки",
    "color": "Колормікс: графіт, коричнево-сірий, бежевий",
    "size_mm": [400, 200, 60],
    "mold_inner_mm": [398, 198],
    "corner_radius_mm": 5,
    "pcs_per_m2": 12.5,
    "kg_per_m2_est": 138,
}

SIZES = {"H": (L, W), "V": (W, L)}


def key_of(w, h, ang=0):
    return "H" if (w > h) == (round(ang) % 180 in (0, 45)) else "V"


def stone(cx, cy, w, h, ang=0):
    return (cx, cy, w, h, ang, key_of(w, h, ang))


def rows(px, spec):
    """spec: список (зсув, кількість) — горизонтальні ряди плит 400x200."""
    out = []
    for r, (offset, n) in enumerate(spec):
        assert n * L == px
        for k in range(n):
            out.append(stone((offset + k * L + L / 2) % px, r * W + W / 2, L, W))
    return px, W * len(spec), out


def running_half():
    return rows(400, [(0, 1), (200, 1)])


def running_third():
    return rows(400, [(0, 1), (400 / 3, 1), (800 / 3, 1)])


def stack():
    return rows(400, [(0, 1)])


def _lattice(cell, v1, v2, px, py, n=12):
    out = []
    for a in range(-n, n + 1):
        for b in range(-n, n + 1):
            ox, oy = a * v1[0] + b * v2[0], a * v1[1] + b * v2[1]
            for cx, cy, w, h, ang in cell:
                x, y = cx + ox, cy + oy
                if 0 <= x < px - 1e-6 and 0 <= y < py - 1e-6:
                    out.append(stone(x, y, w, h, ang))
    return px, py, out


def _rot(cell, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [(c * x - s * y, s * x + c * y, w, h, ang + deg) for x, y, w, h, ang in cell]


def herringbone90():
    # пара: горизонтальна [0,L]x[0,W] + вертикальна [L,L+W]x[W-L,W]
    cell = [(L / 2, W / 2, L, W, 0), (L + W / 2, W - L / 2, W, L, 0)]
    return _lattice(cell, (W, W), (L, -L), 800, 800)


def herringbone45():
    cell = [(L / 2, W / 2, L, W, 0), (L + W / 2, W - L / 2, W, L, 0)]
    r = math.sqrt(2)
    px, py, st = _lattice(_rot(cell, 45), (0, W * r), (L * r, 0), L * r, W * r, n=8)
    return px, py, st


def running45():
    cell = [(L / 2, W / 2, L, W, 0)]
    r = math.sqrt(2)
    s = 200 * r  # період після повороту ложкової перев'язки на 45°
    return _lattice(_rot(cell, 45), (s, s), (0, s), s, s, n=10)


def basketweave():
    out = []
    for bi in range(2):
        for bj in range(2):
            ox, oy = bi * 400, bj * 400
            for k in range(2):
                if (bi + bj) % 2 == 0:
                    out.append(stone(ox + 200, oy + k * 200 + 100, L, W))
                else:
                    out.append(stone(ox + k * 200 + 100, oy + 200, W, L))
    return 800, 800, out


def t_weave():
    """Т-плетінка: блоки 400x600 — поздовжня плита над парою поперечних (Т),
    поруч перевернутий блок (⊥). Поздовжні плити йдуть зигзагом."""
    out = []
    for k in range(2):
        ox = k * 400
        top = k == 0
        yh = 100 if top else 500           # поздовжня плита: зверху або знизу
        yv = 400 if top else 200           # центр пари поперечних
        out.append(stone(ox + 200, yh, L, W))
        out += [stone(ox + 100 + j * 200, yv, W, L) for j in range(2)]
    return 800, 600, out


def cross_band():
    """Два ряди ложкової перев'язки + смуга плит, покладених упоперек."""
    out = [stone(200, 100, L, W), stone(0, 300, L, W)]
    for k in range(2):
        out.append(stone(k * 200 + 100, 400 + 200, W, L))
    return 400, 800, out


def pinwheel():
    """Вітряк: лопаті 800x400 із чотирьох поперечних плит навколо пари 400x400."""
    A, B = 800, 400          # лопать
    c = A - B                # центр 400x400
    cell = [(200, 100, L, W, 0), (200, 300, L, W, 0)]                      # центр: дві горизонтальні
    cell += [(k * 200 + 100, -B / 2, W, L, 0) for k in range(4)]            # верхня лопать [0,800]x[-400,0]
    cell += [(c + 200, k * 200 + 100, L, W, 0) for k in range(4)]           # права лопать [400,800]x[0,800]
    return _lattice(cell, (A, B), (-B, A), 2000, 2000)


SCHEMES = [
    dict(id="01_running_half", name="Ложкова ½", fn=running_half,
         note="Ряди зі зсувом на пів плити, як на фото. Найпростіша у викладці, мінімум підрізки.",
         use="Тротуари, тераси, двори"),
    dict(id="02_running_third", name="Ложкова ⅓", fn=running_third,
         note="Кожен ряд зсунутий на третину плити. Шви утворюють пологу діагональ, менше «смугастості».",
         use="Тротуари, тераси"),
    dict(id="03_stack", name="Сітка", fn=stack,
         note="Прямі наскрізні шви в обох напрямках. Сучасний строгий вигляд, вимагає точної основи.",
         use="Тераси, зони біля будинку"),
    dict(id="04_herringbone90", name="Ялинка 90°", fn=herringbone90,
         note="Класична ялинка 2:1. Найкраще розподіляє навантаження серед усіх схем.",
         use="Двори, під'їзди"),
    dict(id="05_herringbone45", name="Ялинка 45°", fn=herringbone45,
         note="Ялинка під кутом 45° до бордюру. Найвиразніша, але найбільше підрізки біля країв.",
         use="Під'їзди, вхідні зони"),
    dict(id="06_running45", name="Ложкова 45°", fn=running45,
         note="Ложкова перев'язка, повернута на 45°. Візуально розширює вузький простір.",
         use="Доріжки, майданчики"),
    dict(id="07_basketweave", name="Плетінка", fn=basketweave,
         note="Пари плит утворюють квадрати 400×400, що чергуються горизонтально й вертикально.",
         use="Тераси, патіо"),
    dict(id="08_t_weave", name="Т-плетінка", fn=t_weave,
         note="Блоки «Т» і «⊥» чергуються: поздовжня плита то над, то під парою поперечних, утворюючи зигзаг.",
         use="Доріжки, тераси"),
    dict(id="09_cross_band", name="Поперечна смуга", fn=cross_band,
         note="Два ряди ложкової перев'язки, далі смуга з плит упоперек. Ритм для довгих доріжок.",
         use="Довгі доріжки, тротуари"),
    dict(id="10_pinwheel", name="Вітряк", fn=pinwheel,
         note="Лопаті з чотирьох поперечних плит обертаються навколо квадрата з двох плит.",
         use="Площі, акцентні зони"),
]


def build(s):
    return s["fn"]()


def corners(st):
    cx, cy, w, h, ang, _ = st
    a = math.radians(ang)
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)
            for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


def validate(px, py, stones, step=5.0):
    """Перевіряє, що період покритий рівно один раз (без щілин і накладань)."""
    import numpy as np
    X, Y = np.meshgrid(np.arange(0, px, step) + step * 0.371, np.arange(0, py, step) + step * 0.613)
    cnt = np.zeros_like(X, dtype=int)
    for cx, cy, w, h, ang, _ in stones:
        a = math.radians(ang)
        ca, sa = math.cos(a), math.sin(a)
        for sx in (-px, 0, px):
            for sy in (-py, 0, py):
                dx, dy = X - (cx + sx), Y - (cy + sy)
                u, v = dx * ca + dy * sa, -dx * sa + dy * ca
                cnt += (np.abs(u) < w / 2) & (np.abs(v) < h / 2)
    area = sum(w * h for _, _, w, h, _, _ in stones)
    return int(cnt.min()), int(cnt.max()), area / (px * py)


def stats(px, py, stones):
    m2 = px * py / 1e6
    n_h = sum(1 for s in stones if s[5] == "H")
    return {"pcs_per_m2": round(len(stones) / m2, 2),
            "share_H": round(n_h / len(stones), 3)}


if __name__ == "__main__":
    for s in SCHEMES:
        px, py, st = build(s)
        lo, hi, fill = validate(px, py, st)
        print(f"{s['id']:20s} period {px:7.1f} x {py:7.1f}  plates {len(st):3d}  cover {lo}..{hi}  fill {fill:.4f}", stats(px, py, st))
