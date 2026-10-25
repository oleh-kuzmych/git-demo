"""Колормікс з якісного взірця: нові кольори, нова форма плити, нові плями.

Як роблять колормікс: лицьовий шар формують з 2–4 по-різному пігментованих
бетонних сумішей, змішаних не до кінця; ящик-живильник ходить над формою
вперед-назад і розтягує плями смугами вздовж свого ходу; суміш поступово
перемішується від циклу до циклу, тому кожен камінь має власний малюнок і тон.

Генератор повторює це так:
  1. зі взірця (вид строго згори, сірий) вирізає окремі камені; яскравість —
     «карта пігменту»; кожен камінь розвертає смугами по горизонталі;
  2. для кожної плити збирає НОВИЙ малюнок методом image quilting
     (шматки взірця стикуються по шву з найменшою помилкою) — структура
     справжня, але жодна плита не повторює іншу; кожна плита бере
     1–2 «батьківські» камені, тому тон між плитами різниться, як у житті;
  3. перефарбовує карту пігменту в нову палітру: темні смуги, середні тони
     (розділені плавною маскою на теплі й нейтральні) та світлі прожилки;
  4. складає безшовний тайл з потрібною розкладкою.

Запуск: python3 colormix_from_exemplar.py [--palette peach|autumn|terracotta|all] [--scheme 01_running_half] [--seed 1]
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
EXEMPLAR = os.path.join(ROOT, "reference", "colormix_exemplar_grey.webp")
PX_PER_MM = 1.5          # масштаб взірця: камінь ~290 px ≈ 200 мм
JOINT_MM, CORNER_MM = 4.0, 5.0

# Палітри: (позиція 0..1 у карті пігменту, нейтральний колір, теплий колір) — sRGB
PALETTES = {
    # кольори виміряні з ваших фото колорміксу (персикові/сірі зони)
    "peach": [(0.00, (86, 85, 89), (100, 91, 88)), (0.30, (128, 125, 126), (146, 129, 120)),
              (0.62, (176, 168, 165), (196, 173, 159)), (1.00, (222, 214, 206), (230, 212, 196))],
    "autumn": [(0.00, (66, 62, 60), (84, 64, 52)), (0.30, (114, 104, 96), (138, 106, 80)),
               (0.62, (168, 154, 136), (186, 150, 110)), (1.00, (214, 202, 182), (224, 198, 160))],
    "terracotta": [(0.00, (60, 60, 63), (82, 58, 52)), (0.30, (106, 102, 102), (136, 92, 76)),
                   (0.62, (162, 154, 150), (178, 126, 104)), (1.00, (214, 206, 198), (218, 182, 160))],
}


# ------------------------------------------------------------------ взірець
def load_stones(path=EXEMPLAR, margin=16):
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    g = im.mean(2)
    def joints(profile):
        n = len(profile)
        mins = [i for i in range(1, n - 1) if profile[i] < 60 and profile[i] <= profile[i - 1] and profile[i] <= profile[i + 1]]
        return [0] + mins + [n - 1]
    xs, ys = joints(g.mean(0)), joints(g.mean(1))
    stones = []
    for y0, y1 in zip(ys[:-1], ys[1:]):
        for x0, x1 in zip(xs[:-1], xs[1:]):
            c = g[y0 + margin:y1 - margin, x0 + margin:x1 - margin]
            if min(c.shape) < 100:
                continue
            gx, gy = cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=5), cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=5)
            if (gy * gy).mean() < (gx * gx).mean():   # градієнт переважно по X → смуги вертикальні
                c = np.ascontiguousarray(np.rot90(c))
            stones.append(c)
    return stones


# ------------------------------------------------------------------ image quilting
def _min_cut(err, axis):
    """Шов мінімальної помилки. err: (h, w). axis=1 — вертикальний шов (ліве перекриття)."""
    e = err if axis == 1 else err.T
    h, w = e.shape
    cost = e.copy()
    for i in range(1, h):
        prev = cost[i - 1]
        left = np.r_[np.inf, prev[:-1]]
        right = np.r_[prev[1:], np.inf]
        cost[i] += np.minimum(np.minimum(left, prev), right)
    mask = np.zeros_like(e, bool)              # True — брати новий патч
    j = int(np.argmin(cost[-1]))
    for i in range(h - 1, -1, -1):
        mask[i, j:] = True
        if i:
            lo, hi = max(0, j - 1), min(w, j + 2)
            j = lo + int(np.argmin(cost[i - 1, lo:hi]))
    return mask if axis == 1 else mask.T


def quilt(h, w, sources, rng, P=112, O=32, ncand=400):
    step = P - O
    ny, nx = math.ceil((h - O) / step), math.ceil((w - O) / step)
    H, W = ny * step + O, nx * step + O
    out = np.zeros((H, W), np.float32)
    wins = [np.lib.stride_tricks.sliding_window_view(s, (P, P)) for s in sources]
    for iy in range(ny):
        for ix in range(nx):
            y, x = iy * step, ix * step
            k = rng.integers(len(wins)); wv = wins[k]
            cy = rng.integers(0, wv.shape[0], ncand); cx = rng.integers(0, wv.shape[1], ncand)
            cand = wv[cy, cx]                                   # (n, P, P)
            if iy == 0 and ix == 0:
                out[:P, :P] = cand[0]; continue
            err = np.zeros(ncand, np.float32)
            if ix > 0:
                err += ((cand[:, :, :O] - out[y:y + P, x:x + O]) ** 2).sum((1, 2))
            if iy > 0:
                err += ((cand[:, :O, :] - out[y:y + O, x:x + P]) ** 2).sum((1, 2))
            best = np.flatnonzero(err <= err.min() * 1.08 + 1e-6)
            pch = cand[rng.choice(best)].copy()
            m = np.ones((P, P), bool)
            if ix > 0:
                m[:, :O] &= _min_cut((pch[:, :O] - out[y:y + P, x:x + O]) ** 2, 1)
            if iy > 0:
                m[:O, :] &= _min_cut((pch[:O, :] - out[y:y + O, x:x + P]) ** 2, 0)
            region = out[y:y + P, x:x + P]
            region[m] = pch[m]
    return out[:h, :w]


# ------------------------------------------------------------------ колір
def colorize(L, warm, palette, lo, hi):
    t = np.clip((L - lo) / (hi - lo), 0, 1)
    pos = np.array([p[0] for p in palette])
    neu = np.array([p[1] for p in palette], np.float32); wrm = np.array([p[2] for p in palette], np.float32)
    out = np.zeros(L.shape + (3,), np.float32)
    for ch in range(3):
        a = np.interp(t, pos, neu[:, ch]); b = np.interp(t, pos, wrm[:, ch])
        out[..., ch] = a * (1 - warm) + b * warm
    return out


def aniso_noise(shape, sx, sy, rng):
    n = rng.standard_normal(shape).astype(np.float32)
    n = gaussian_filter(n, (sy, sx), mode="wrap")
    return (n - n.mean()) / (n.std() + 1e-9)


# ------------------------------------------------------------------ тайл
def render_tile(scheme_id, palette_name, seed=1, stones=None):
    scheme = next(s for s in SCHEMES if s["id"] == scheme_id)
    px, py, plates = build(scheme)
    n = max(1, round(1600 / px)); m = max(1, round(1600 / py))
    ex, ey = n * px, m * py
    W, H = round(ex * PX_PER_MM), round(ey * PX_PER_MM)
    stones = stones or load_stones()
    allv = np.concatenate([s.ravel() for s in stones])
    lo, hi = np.percentile(allv, 1), np.percentile(allv, 99.5)
    pal = PALETTES[palette_name]
    img = np.zeros((H, W, 3), np.float32) + np.array([30, 29, 29], np.float32)   # шви
    j2, cr = JOINT_MM / 2, CORNER_MM
    cache = {}
    for i in range(-1, n + 1):
        for j in range(-1, m + 1):
            for cx0, cy0, w, h, ang, _ in plates:
                cx, cy = cx0 + i * px, cy0 + j * py
                if not (-300 < cx < ex + 300 and -300 < cy < ey + 300):
                    continue
                key = (round(cx % ex, 1) % round(ex, 1), round(cy % ey, 1) % round(ey, 1))
                if key not in cache:
                    r = np.random.default_rng(int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest()[:8], 16))
                    Lmm, Smm = max(w, h), min(w, h)
                    pw, ph = int(Lmm * PX_PER_MM) + 8, int(Smm * PX_PER_MM) + 8
                    src = [stones[q] for q in r.choice(len(stones), size=r.integers(1, 3), replace=False)]
                    L = quilt(ph, pw, src, r)
                    L += r.normal(0, 4)                                     # дрейф суміші між циклами
                    warm = np.clip(0.5 + 0.42 * aniso_noise(L.shape, 80, 12, r) + r.normal(0, 0.18), 0, 1)
                    Lc = gaussian_filter(L, 1.6)                           # колір — зі згладженої карти
                    rgb = colorize(Lc, gaussian_filter(warm, 3), pal, lo, hi)
                    rgb *= (1 + (L - Lc) / 255.0 * 1.1)[..., None]          # зерно — лише світлотінню
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
                if w < h:                      # довга сторона плити = напрям смуг
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
                rim = np.clip(-d / 5.0, 0, 1)                                   # фаска: легке затемнення краю
                col = col * (0.80 + 0.20 * rim ** 0.6)[..., None]
                sl = (slice(y0, y1), slice(x0, x1))
                img[sl] = img[sl] * (1 - cov[..., None]) + col * cov[..., None]
    # дрібна пористість поверхні (взірець уже має зерно; додаємо ледь-ледь)
    rng = np.random.default_rng(seed)
    img += rng.normal(0, 2.0, img.shape[:2])[..., None]
    return np.clip(img, 0, 255).astype(np.uint8), (ex, ey)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--palette", default="peach")
    ap.add_argument("--scheme", default="01_running_half")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    stones = load_stones()
    print("каменів у взірці:", len(stones))
    names = list(PALETTES) if a.palette == "all" else [a.palette]
    for pn in names:
        img, (ex, ey) = render_tile(a.scheme, pn, a.seed, stones)
        out = os.path.join(ROOT, "colormix", f"colormix_{pn}_{a.scheme}_{int(ex)}x{int(ey)}mm.jpg")
        Image.fromarray(img).save(out, quality=95)
        print(out, img.shape)


if __name__ == "__main__":
    main()
