"""Колормікс «мазком»: кожна плита — окремий процес формування, без копіювання.

Як роблять колормікс: лицьовий шар складають з 3–4 по-різному пігментованих
сумішей, змішаних не до кінця (згустки); ящик-живильник одним ходом протягує
суміш над формою — згустки витягуються в плавні смуги вздовж ходу, з
розмазаними кінцями; від циклу до циклу суміш перемішується, тож пропорції
кольорів і малюнок у кожного каменя свої.

Модель для кожної плити:
  1. згустки пігментів — поля концентрацій (м'який softmax шумів різного
     масштабу) з власними пропорціями плити;
  2. мазок — поля розтягуються вздовж ходу живильника (анізотропія), траєкторія
     злегка хвиляста, ширина й довжина мазка змінюються поперек плити,
     передній край змазаний (спрямоване розмиття);
  3. колір = суміш пігментів; зверху — дрібні волокна й зерна піску.
Усі плити різні: кожна має власне зерно генератора.

Запуск: python3 colormix_strokes.py [--palette peach|autumn|terracotta|grey|all] [--scheme 01_running_half] [--seed 1] [--seamless]
"""
import argparse
import hashlib
import math
import os

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from schemes import SCHEMES, build

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PX_PER_MM = 1.5
JOINT_MM, CORNER_MM = 4.0, 5.0

# пігменти: (колір sRGB, середня частка в суміші)
PALETTES = {
    "grey":       [((214, 212, 208), 0.38), ((174, 173, 170), 0.30), ((136, 136, 138), 0.24), ((100, 101, 104), 0.08)],
    # з ваших фото колорміксу
    "peach":      [((222, 211, 199), 0.32), ((198, 170, 151), 0.34), ((143, 139, 139), 0.25), ((104, 100, 102), 0.09)],
    "autumn":     [((218, 202, 176), 0.32), ((188, 150, 110), 0.34), ((132, 116, 100), 0.25), ((88, 80, 74), 0.09)],
    "terracotta": [((216, 202, 188), 0.30), ((178, 116, 90), 0.36), ((138, 130, 128), 0.25), ((90, 84, 84), 0.09)],
}


def aniso(shape, sx, sy, rng):
    """Анізотропний гаусів шум, std = 1 (sx — уздовж ходу живильника)."""
    n = gaussian_filter(rng.standard_normal(shape).astype(np.float32), (sy, sx), mode="reflect")
    return (n - n.mean()) / (n.std() + 1e-9)


def plate_field(h, w, palette, rng):
    """Колір однієї плити (h x w px), хід живильника — уздовж осі X."""
    K = len(palette)
    pad = 24
    H, W = h + 2 * pad, w + 2 * pad
    # 2. мазок: хвиляста траєкторія та змінний зсув рядків
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wave = aniso((H, W), 160, 120, rng) * rng.uniform(3, 9)              # плавний вигин мазка
    shear = aniso((H, 1), 1, rng.uniform(10, 30), rng)[:, :1] * rng.uniform(10, 40)
    mx = xx + shear
    my = yy + wave
    # 1. згустки пігментів із власними пропорціями плити (дрейф суміші)
    props = np.array([p for _, p in palette]) * rng.lognormal(0, 0.5, K)
    props /= props.sum()
    beta = rng.uniform(1.7, 2.5)                                        # наскільки суміш «неперемішана»
    sx_streak = rng.uniform(9, 18)                                      # довжина смуг мазка, px
    logits = []
    for k in range(K):
        zone = aniso((H, W), rng.uniform(45, 85), rng.uniform(16, 30), rng)        # великі «хмари» одного кольору
        streak = aniso((H, W), sx_streak * rng.uniform(0.8, 1.25), rng.uniform(1.5, 3.0), rng)  # смуги мазка
        micro = aniso((H, W), rng.uniform(5, 9), rng.uniform(0.6, 1.0), rng)       # дрібні волокна
        n = 0.85 * zone + 0.55 * streak + 0.30 * micro
        n = cv2.remap(n, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        logits.append(beta * n + math.log(props[k]))
    lg = np.stack(logits)
    c = np.exp(lg - lg.max(0)); c /= c.sum(0)                          # плавний склад суміші
    # зерна: кожна піщинка вибирає пігмент випадково з імовірністю c (Gumbel-вибір, м'який)
    G = -np.log(-np.log(np.clip(rng.random((K, H, W)), 1e-6, 1 - 1e-6))).astype(np.float32)
    G = np.stack([cv2.GaussianBlur(g, (0, 0), 0.75) for g in G]) * 1.9
    q = (np.log(c + 1e-6) + G) / 0.6
    pg = np.exp(q - q.max(0)); pg /= pg.sum(0)
    f = 0.45 * pg + 0.55 * c
    cols = np.array([c for c, _ in palette], np.float32) * rng.normal(1, 0.025, (K, 1))
    rgb = np.tensordot(f.transpose(1, 2, 0), cols, axes=([2], [0]))
    rgb = cv2.GaussianBlur(rgb, (0, 0), 0.55)
    # 3. шорсткість поверхні: зерна піску, пори, світлі піщинки кварцу (поверх кольору)
    g1 = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), 0.95); g1 /= g1.std() + 1e-9
    g2 = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), 2.2); g2 /= g2.std() + 1e-9
    rgb *= (1 + 0.065 * g1 + 0.04 * g2)[..., None]
    sp = gaussian_filter(rng.random((H, W)).astype(np.float32), 0.5)
    hi_q, lo_q = np.quantile(sp, 0.992), np.quantile(sp, 0.006)
    rgb[sp > hi_q] = rgb[sp > hi_q] * 0.45 + 125          # світлі піщинки кварцу
    rgb[sp < lo_q] = rgb[sp < lo_q] * 0.62                # пори
    return rgb[pad:pad + h, pad:pad + w]


def render(scheme_id, palette_name, seed=1, seamless=False, size_mm=1600):
    scheme = next(s for s in SCHEMES if s["id"] == scheme_id)
    px, py, plates = build(scheme)
    if seamless:
        n, m = max(1, round(size_mm / px)), max(1, round(size_mm / py))
        ex, ey = n * px, m * py
    else:
        ex = ey = size_mm
        n, m = math.ceil(ex / px), math.ceil(ey / py)
    W, H = round(ex * PX_PER_MM), round(ey * PX_PER_MM)
    pal = PALETTES[palette_name]
    img = np.zeros((H, W, 3), np.float32) + np.array([30, 29, 29], np.float32)
    j2, cr = JOINT_MM / 2, CORNER_MM
    cache = {}
    ox, oy = (0, 0) if seamless else (-137.0, -61.0)     # без тайлу — розкладка з довільного місця
    for i in range(-1, n + 2):
        for j in range(-1, m + 2):
            for cx0, cy0, w, h, ang, _ in plates:
                cx, cy = cx0 + i * px + ox, cy0 + j * py + oy
                if not (-300 < cx < ex + 300 and -300 < cy < ey + 300):
                    continue
                key = (round(cx % ex, 1) % round(ex, 1), round(cy % ey, 1) % round(ey, 1)) if seamless else (round(cx, 1), round(cy, 1))
                if key not in cache:
                    r = np.random.default_rng(int(hashlib.md5(f"{seed}:{palette_name}:{key}".encode()).hexdigest()[:12], 16))
                    Lmm, Smm = max(w, h), min(w, h)
                    rgb = plate_field(int(Smm * PX_PER_MM) + 8, int(Lmm * PX_PER_MM) + 8, pal, r)
                    if r.random() < 0.5:
                        rgb = rgb[::-1, ::-1]
                    cache[key] = rgb
                rgb = cache[key]
                a = math.radians(ang); ca, sa = math.cos(a), math.sin(a)
                hw = abs(w / 2 * ca) + abs(h / 2 * sa); hh = abs(w / 2 * sa) + abs(h / 2 * ca)
                x0, x1 = max(0, int((cx - hw) * W / ex)), min(W, int((cx + hw) * W / ex) + 1)
                y0, y1 = max(0, int((cy - hh) * H / ey)), min(H, int((cy + hh) * H / ey) + 1)
                if x0 >= x1 or y0 >= y1:
                    continue
                X, Y = np.meshgrid((np.arange(x0, x1) + 0.5) * ex / W - cx, (np.arange(y0, y1) + 0.5) * ey / H - cy)
                u, v = X * ca + Y * sa, -X * sa + Y * ca
                if w < h:
                    u, v = v, -u
                    Lw, Sw = h, w
                else:
                    Lw, Sw = w, h
                qx, qy = np.abs(u) - (Lw / 2 - j2 - cr), np.abs(v) - (Sw / 2 - j2 - cr)
                d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - cr
                cov = np.clip(-d * PX_PER_MM + 0.5, 0, 1)
                if not cov.any():
                    continue
                sx_ = np.clip(((u + Lw / 2) * PX_PER_MM + 4).astype(int), 0, rgb.shape[1] - 1)
                sy_ = np.clip(((v + Sw / 2) * PX_PER_MM + 4).astype(int), 0, rgb.shape[0] - 1)
                col = rgb[sy_, sx_]
                rim = np.clip(-d / 5.0, 0, 1)
                col = col * (0.80 + 0.20 * rim ** 0.6)[..., None]
                sl = (slice(y0, y1), slice(x0, x1))
                img[sl] = img[sl] * (1 - cov[..., None]) + col * cov[..., None]
    return np.clip(img, 0, 255).astype(np.uint8), (ex, ey)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--palette", default="peach")
    ap.add_argument("--scheme", default="01_running_half")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seamless", action="store_true", help="безшовний тайл для текстур (крайні плити повторюються з іншого боку)")
    a = ap.parse_args()
    names = [p for p in PALETTES if p != "grey"] if a.palette == "all" else [a.palette]
    for pn in names:
        img, (ex, ey) = render(a.scheme, pn, a.seed, a.seamless)
        tag = "tile" if a.seamless else "unique"
        out = os.path.join(ROOT, "colormix", f"colormix_{pn}_{a.scheme}_{int(ex)}x{int(ey)}mm_{tag}.jpg")
        Image.fromarray(img).save(out, quality=95)
        print(out, img.shape)


if __name__ == "__main__":
    main()
