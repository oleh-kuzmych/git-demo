"""Безшовні PBR-текстури для кожної схеми.

Карти (OpenGL-нормалі, як у Blender):
  *_BaseColor.jpg  *_Normal.jpg  *_Roughness.jpg  *_AO.jpg  *_Displacement.png (16 біт, 0..1 = 0..10 мм)

Запуск: python3 build_textures.py [--res 2048] [--only 06_herringbone90]
"""
import argparse
import hashlib
import json
import math
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from schemes import SCHEMES, build

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JOINT = 4.0      # ширина шва, мм
CORNER = 5.0     # радіус заокруглення кута в плані, мм
BEVEL = 6.0      # фаска (заокруглення ребра), мм
TOP = 8.0        # висота верху каменю над дном шва, мм
SAND = 2.6       # рівень піску в шві, мм
DISP_RANGE = 10.0


def texture_extent(px, py):
    best = None
    for n in range(1, 16):
        for m in range(1, 16):
            ex, ey = n * px, m * py
            if not (850 <= ex <= 1700 and 850 <= ey <= 1700):
                continue
            score = abs(ex - ey) / max(ex, ey) + 0.15 * abs(max(ex, ey) - 1200) / 1200
            if best is None or score < best[0]:
                best = (score, n, m)
    _, n, m = best
    return n, m


def pnoise(shape, wl_px, rng):
    """Періодичний (безшовний) шум із довжиною хвилі wl_px, std = 1."""
    h, w = shape
    white = rng.standard_normal(shape)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    sigma = 1.0 / max(wl_px, 1.0)
    filt = np.exp(-(f / sigma) ** 2)
    out = np.fft.irfft2(np.fft.rfft2(white) * filt, s=shape)
    out -= out.mean()
    return (out / (out.std() + 1e-9)).astype(np.float32)


def stone_rng(x, y, ex, ey):
    key = f"{round(x % ex, 1) % round(ex, 1)}:{round(y % ey, 1) % round(ey, 1)}"
    return np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))


def render(scheme, res):
    px, py, stones = build(scheme)
    n, m = texture_extent(px, py)
    ex, ey = n * px, m * py
    ppm = res / max(ex, ey)
    W, H = round(ex * ppm), round(ey * ppm)
    sx, sy = W / ex, H / ey  # пікселів на мм по осях (майже однакові)
    rng = np.random.default_rng(int(hashlib.md5(scheme["id"].encode()).hexdigest()[:8], 16))

    prof = np.zeros((H, W), np.float32)     # 0..1 профіль ребра
    top = np.zeros((H, W), np.float32)      # висота верху каменю (з нахилом)
    tone = np.zeros((H, W), np.float32)     # тон каменю
    cov = np.zeros((H, W), np.float32)      # покриття каменем (антиаліасинг)

    reach = 200
    for i in range(-1, n + 2):
        for j in range(-1, m + 2):
            for cx0, cy0, w, h, ang, _ in stones:
                cx, cy = cx0 + i * px, cy0 + j * py
                if not (-reach < cx < ex + reach and -reach < cy < ey + reach):
                    continue
                a = math.radians(ang)
                ca, sa = math.cos(a), math.sin(a)
                hw = abs(w / 2 * ca) + abs(h / 2 * sa) + 2
                hh = abs(w / 2 * sa) + abs(h / 2 * ca) + 2
                x0, x1 = max(0, int((cx - hw) * sx)), min(W, int((cx + hw) * sx) + 1)
                y0, y1 = max(0, int((cy - hh) * sy)), min(H, int((cy + hh) * sy) + 1)
                if x0 >= x1 or y0 >= y1:
                    continue
                X = (np.arange(x0, x1) + 0.5) / sx - cx
                Y = (np.arange(y0, y1) + 0.5) / sy - cy
                X, Y = np.meshgrid(X, Y)
                u = X * ca + Y * sa
                v = -X * sa + Y * ca
                r = stone_rng(cx, cy, ex, ey)
                # легка неправильність форми: різні фаски та радіуси
                bev = BEVEL * r.uniform(0.8, 1.3)
                cr = CORNER * r.uniform(0.8, 1.4)
                ax, ay = w / 2 - JOINT / 2 - cr, h / 2 - JOINT / 2 - cr
                qx, qy = np.abs(u) - ax, np.abs(v) - ay
                d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - cr
                t = np.clip(-d / bev, 0, 1)
                p = np.sqrt(1 - (1 - t) ** 2)                      # заокруглене ребро
                c = np.clip(-d * sx + 0.5, 0, 1)
                gx, gy = r.normal(0, 0.0035, 2)
                tp = TOP + r.normal(0, 0.25) + gx * u + gy * v
                sl = (slice(y0, y1), slice(x0, x1))
                upd = c > cov[sl]
                prof[sl] = np.where(upd, p, prof[sl])
                top[sl] = np.where(upd, tp, top[sl])
                tone[sl] = np.where(upd, r.normal(0, 1), tone[sl])
                cov[sl] = np.maximum(cov[sl], c)

    mm = lambda v: v * ppm
    warp = pnoise((H, W), mm(45), rng)
    mid = pnoise((H, W), mm(12), rng)
    fine = pnoise((H, W), mm(0.9), rng)
    fine2 = pnoise((H, W), mm(0.6), rng)
    pores = np.clip(pnoise((H, W), mm(0.7), rng) - 2.6, 0, None)
    aggr = pnoise((H, W), mm(1.6), rng)

    stone_top = top + 0.30 * warp + 0.10 * mid + 0.05 * fine - 0.35 * pores
    stone_h = 1.8 + (stone_top - 1.8) * prof
    sand_h = SAND + 0.30 * fine2 + 0.25 * mid
    height = np.where(cov > 0, np.maximum(stone_h, sand_h * (1 - cov) + stone_h * cov), sand_h)
    height = np.maximum(height, sand_h)

    # --- колір (sRGB)
    base = np.array([127, 129, 132], np.float32)
    lum = 1 + 0.04 * tone + 0.018 * warp + 0.015 * mid
    stone_rgb = base[None, None, :] * lum[..., None]
    stone_rgb += (7.5 * fine)[..., None]
    stone_rgb += (np.where(aggr > 1.9, 16, 0) - np.where(aggr < -2.1, 14, 0))[..., None]
    stone_rgb += (7 * (1 - prof) * (cov > 0))[..., None]   # потерті світліші ребра
    stone_rgb -= (35 * pores)[..., None]
    sand_rgb = np.array([164, 159, 149], np.float32)[None, None, :] * (1 + 0.08 * fine2 + 0.04 * mid)[..., None]
    sand_rgb -= (np.where(fine > 2.2, 30, 0))[..., None]
    a = cov[..., None]
    rgb = stone_rgb * a + sand_rgb * (1 - a)

    # --- AO
    hb1 = gaussian_filter(height, sigma=mm(2.5), mode="wrap")
    hb2 = gaussian_filter(height, sigma=mm(8), mode="wrap")
    cav = np.maximum(hb1 - height, 0) * 0.18 + np.maximum(hb2 - height, 0) * 0.10
    ao = np.clip(1 - cav, 0.35, 1)
    rgb *= (0.75 + 0.25 * ao)[..., None]

    # --- roughness
    rough = (0.80 + 0.05 * mid + 0.03 * fine) * cov + (0.95 + 0.02 * fine2) * (1 - cov)
    rough -= 0.08 * (1 - prof) * cov

    # --- нормалі (OpenGL, +Y вгору)
    dhdx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * sx / 2
    dhdr = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * sy / 2
    nrm = np.stack([-dhdx, dhdr, np.ones_like(height)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)

    out = f"{ROOT}/textures/{scheme['id']}"
    os.makedirs(out, exist_ok=True)
    pre = f"{out}/{scheme['id']}"
    u8 = lambda x: np.clip(x, 0, 255).astype(np.uint8)
    Image.fromarray(u8(rgb)).save(f"{pre}_BaseColor.jpg", quality=92)
    Image.fromarray(u8((nrm + 1) * 127.5)).save(f"{pre}_Normal.jpg", quality=93)
    Image.fromarray(u8(rough * 255)).save(f"{pre}_Roughness.jpg", quality=90)
    Image.fromarray(u8(ao * 255)).save(f"{pre}_AO.jpg", quality=90)
    disp = np.clip(height / DISP_RANGE, 0, 1)
    Image.fromarray((disp * 65535).astype(np.uint16)).save(f"{pre}_Displacement.png", optimize=True)
    return {"texture_mm": [round(ex, 3), round(ey, 3)], "texture_px": [W, H], "repeats": [n, m]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--only")
    args = ap.parse_args()
    meta_path = f"{ROOT}/textures/textures.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for s in SCHEMES:
        if args.only and s["id"] != args.only:
            continue
        meta[s["id"]] = render(s, args.res)
        meta[s["id"]]["displacement_range_mm"] = DISP_RANGE
        print(s["id"], meta[s["id"]], flush=True)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    json.dump(meta, open(meta_path, "w"), indent=1)


if __name__ == "__main__":
    main()
