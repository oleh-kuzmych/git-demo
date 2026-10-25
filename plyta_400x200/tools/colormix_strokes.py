"""Колормікс, як на реальних фото (reference/real_*.jpg).

Що видно на справжніх каменях і як це відтворено:
  * смуги в усіх каменів ідуть в ОДНОМУ напрямку — уздовж ходу живильника
    (а не вздовж довгої сторони каменя): параметр --dir;
  * камінь переважно одного-двох тонів; між тонами — широкий м'який перехід
    поперек ходу (напр. половина бежева, половина сіра);
  * поверх — тонкі переривчасті смуги інших пігментів, як від сухого пензля;
  * контраст усередині каменю невеликий, основна різниця — між каменями;
  * поверхня гладка: дрібне зерно, поодинокі пори й світлі піщинки.
Кожен камінь генерується окремо зі свого зерна — двох однакових немає.

Запуск: python3 colormix_strokes.py [--palette brown_grey|peach_grey|sand|rose|grey|all]
        [--scheme 01_running_half] [--seed 1] [--dir v|h] [--seamless]
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
JOINT_MM, CORNER_MM = 3.0, 4.0

# пігменти (sRGB) і середня частка; виміряно на reference/real_*.jpg
PALETTES = {
    "brown_grey": [((186, 176, 166), 0.18), ((172, 146, 124), 0.32), ((130, 124, 120), 0.32), ((90, 85, 82), 0.18)],
    "peach_grey": [((188, 180, 176), 0.26), ((174, 154, 144), 0.32), ((140, 140, 145), 0.30), ((108, 106, 110), 0.12)],
    "sand":       [((198, 192, 186), 0.28), ((180, 160, 146), 0.34), ((142, 142, 147), 0.28), ((106, 103, 106), 0.10)],
    "rose":       [((188, 182, 177), 0.34), ((168, 140, 132), 0.36), ((158, 155, 152), 0.24), ((132, 122, 119), 0.06)],
    "grey":       [((198, 199, 194), 0.30), ((166, 168, 164), 0.32), ((136, 138, 136), 0.26), ((104, 106, 106), 0.12)],
}


def aniso(shape, s_across, s_along, rng):
    """Анізотропний шум (std = 1); вісь 0 — уздовж ходу живильника."""
    n = gaussian_filter(rng.standard_normal(shape).astype(np.float32), (s_along, s_across), mode="reflect")
    return (n - n.mean()) / (n.std() + 1e-9)


def stone_field(h, w, palette, rng):
    """Колір каменя h x w px; хід живильника — уздовж осі 0 (вниз)."""
    K = len(palette)
    pad = 20
    H, W = h + 2 * pad, w + 2 * pad
    base_p = np.array([p for _, p in palette])
    # склад суміші цього каменя: 1–2 домінантні пігменти (дрейф суміші між циклами)
    p = base_p * rng.lognormal(0, 0.7, K)
    p /= p.sum()
    streaky = rng.beta(2.0, 2.2)                       # 0 — майже однотонний, 1 — дуже смугастий
    across = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    logits = []
    for k in range(K):
        big = aniso((H, W), rng.uniform(25, 55), rng.uniform(100, 220), rng)     # широкі тонові зони
        ramp = across * rng.normal(0, 0.9)                                       # плавний перехід поперек ходу
        mid = aniso((H, W), rng.uniform(4, 9), rng.uniform(45, 110), rng)        # середні смуги
        fine = aniso((H, W), rng.uniform(0.8, 1.4), rng.uniform(6, 16), rng)     # короткі штрихи «сухий пензель»
        gate = 1 / (1 + np.exp(-2.5 * aniso((H, W), rng.uniform(2, 5), rng.uniform(3, 7), rng)))
        grain = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), 0.8); grain /= grain.std() + 1e-9
        n = 1.1 * big + ramp + (0.25 + streaky) * (0.75 * mid + 0.95 * fine * gate) + (0.25 + 0.4 * streaky) * grain
        logits.append(1.6 * n + math.log(p[k]))
    lg = np.stack(logits)
    f = np.exp(lg - lg.max(0)); f /= f.sum(0)
    cols = np.array([c for c, _ in palette], np.float32) * rng.normal(1, 0.02, (K, 1))
    rgb = np.tensordot(f.transpose(1, 2, 0), cols, axes=([2], [0]))
    rgb *= rng.normal(1, 0.035)                        # загальний тон каменя
    # поверхня: дрібне зерно, легка плямистість, пори, світлі піщинки
    g = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), 0.8); g /= g.std() + 1e-9
    g2 = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), 1.8); g2 /= g2.std() + 1e-9
    mott = aniso((H, W), 6, 8, rng)
    rgb *= (1 + 0.045 * g + 0.03 * g2 + 0.015 * mott)[..., None]
    r = rng.random((H, W))
    pores = gaussian_filter((r < 0.0006).astype(np.float32), 0.6) * 4
    specks = gaussian_filter((r > 0.99965).astype(np.float32), 0.55) * 4
    rgb *= (1 - 0.3 * np.clip(pores, 0, 1))[..., None]
    rgb += (40 * np.clip(specks, 0, 1))[..., None]
    return rgb[pad:pad + h, pad:pad + w]


def render(scheme_id, palette_name, seed=1, seamless=False, size_mm=1600, direction="v"):
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
    img = np.zeros((H, W, 3), np.float32) + np.array([34, 32, 31], np.float32)
    j2, cr = JOINT_MM / 2, CORNER_MM
    ox, oy = (0, 0) if seamless else (-137.0, -61.0)
    cache = {}
    for i in range(-1, n + 2):
        for j in range(-1, m + 2):
            for cx0, cy0, w, h, ang, _ in plates:
                cx, cy = cx0 + i * px + ox, cy0 + j * py + oy
                if not (-300 < cx < ex + 300 and -300 < cy < ey + 300):
                    continue
                a = math.radians(ang); ca, sa = math.cos(a), math.sin(a)
                hw = abs(w / 2 * ca) + abs(h / 2 * sa); hh = abs(w / 2 * sa) + abs(h / 2 * ca)
                key = ((round(cx % ex, 1) % round(ex, 1), round(cy % ey, 1) % round(ey, 1)) if seamless
                       else (round(cx, 1), round(cy, 1)))
                fh, fw = int(2 * hh * PX_PER_MM) + 8, int(2 * hw * PX_PER_MM) + 8
                if key not in cache:
                    r = np.random.default_rng(int(hashlib.md5(f"{seed}:{palette_name}:{key}".encode()).hexdigest()[:12], 16))
                    if direction == "v":
                        cache[key] = stone_field(fh, fw, pal, r)
                    else:
                        cache[key] = np.ascontiguousarray(stone_field(fw, fh, pal, r).transpose(1, 0, 2))
                rgb = cache[key]
                x0, x1 = max(0, int((cx - hw) * W / ex)), min(W, int((cx + hw) * W / ex) + 1)
                y0, y1 = max(0, int((cy - hh) * H / ey)), min(H, int((cy + hh) * H / ey) + 1)
                if x0 >= x1 or y0 >= y1:
                    continue
                X, Y = np.meshgrid((np.arange(x0, x1) + 0.5) * ex / W - cx, (np.arange(y0, y1) + 0.5) * ey / H - cy)
                u, v = X * ca + Y * sa, -X * sa + Y * ca
                qx, qy = np.abs(u) - (w / 2 - j2 - cr), np.abs(v) - (h / 2 - j2 - cr)
                d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - cr
                cov = np.clip(-d * PX_PER_MM + 0.5, 0, 1)
                if not cov.any():
                    continue
                sx_ = np.clip(((X + hw) * PX_PER_MM + 4).astype(int), 0, rgb.shape[1] - 1)
                sy_ = np.clip(((Y + hh) * PX_PER_MM + 4).astype(int), 0, rgb.shape[0] - 1)
                col = rgb[sy_, sx_]
                rim = np.clip(-d / 3.5, 0, 1)                    # фаска: вузький темніший край
                col = col * (0.72 + 0.28 * rim ** 0.5)[..., None]
                sl = (slice(y0, y1), slice(x0, x1))
                img[sl] = img[sl] * (1 - cov[..., None]) + col * cov[..., None]
    return np.clip(img, 0, 255).astype(np.uint8), (ex, ey)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--palette", default="brown_grey")
    ap.add_argument("--scheme", default="01_running_half")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dir", default="v", choices=["v", "h"], help="напрям ходу живильника (смуг)")
    ap.add_argument("--seamless", action="store_true", help="безшовний тайл для текстур")
    a = ap.parse_args()
    names = [p for p in PALETTES if p != "grey"] if a.palette == "all" else [a.palette]
    for pn in names:
        img, (ex, ey) = render(a.scheme, pn, a.seed, a.seamless, direction=a.dir)
        tag = "tile" if a.seamless else "unique"
        out = os.path.join(ROOT, "colormix", f"colormix_{pn}_{a.scheme}_{int(ex)}x{int(ey)}mm_{tag}.jpg")
        Image.fromarray(img).save(out, quality=95)
        print(out, img.shape)


if __name__ == "__main__":
    main()
