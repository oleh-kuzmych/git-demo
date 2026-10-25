"""Безшовні PBR-текстури для кожної схеми (плита 400x200, колормікс).

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

import cv2

SWATCH_DIR = None  # заповнюється в load_swatches()
SWATCHES = []


def load_swatches():
    """Реальні камені з фото колорміксу (reference/swatches, див. extract_swatches.py).

    Кольори розкладаються на головні компоненти (PCA) усіх каменів з фото:
    t1 — головна вісь «персиковий ↔ сірий/графіт», t2, t3 — відтінок.
    Повертає (mu, E, список карт [t1,t2,t3] у 4× збільшенні, список квантилів t1)."""
    d = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "reference", "swatches")
    meta = json.load(open(os.path.join(d, "swatches.json")))
    ims = [cv2.cvtColor(cv2.imread(os.path.join(d, e["file"])), cv2.COLOR_BGR2RGB).astype(np.float32) for e in meta["swatches"]]
    allpx = np.concatenate([im.reshape(-1, 3) for im in ims])
    mu = allpx.mean(0)
    _, _, vt = np.linalg.svd(allpx - mu, full_matrices=False)
    E = vt                                   # рядки — головні компоненти
    maps, quant = [], []
    for im in ims:
        t = (im - mu) @ E.T                  # H x W x 3
        quant.append(np.quantile(t[..., 0], np.linspace(0, 1, 257)).astype(np.float32))
        t = cv2.resize(t, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        maps.append(cv2.GaussianBlur(t, (0, 0), 2.0))
    return mu.astype(np.float32), E.astype(np.float32), maps, quant, meta["zones"]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JOINT = 3.0      # ширина шва, мм
CORNER = 5.0     # радіус заокруглення кута в плані, мм
BEVEL = 5.0      # фаска (заокруглення ребра), мм
TOP = 8.0        # висота верху каменю над дном шва, мм
SAND = 2.6       # рівень піску в шві, мм
DISP_RANGE = 10.0


def texture_extent(px, py):
    best = None
    for n in range(1, 16):
        for m in range(1, 16):
            ex, ey = n * px, m * py
            if not (850 <= ex <= 2500 and 850 <= ey <= 2500):
                continue
            if abs(ex - ey) > 1e-6:  # лише квадратні тайли (вимога Lumion та ін.)
                continue
            score = abs(max(ex, ey) - 1600) / 1600
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
    sid = np.zeros((H, W), np.int32)        # номер плити (0 — шов)
    tmap = np.zeros((H, W, 3), np.float32)  # макро-розкладка плям з фото (PCA-координати)
    axis = np.zeros((H, W), np.int8)        # напрям довгої сторони: 0/45/90/135°
    attrs = [None]
    if not SWATCHES:
        SWATCHES.extend(load_swatches())
    MU, E, SWM, SWQ, ZONES = SWATCHES

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
                p = 1 - (1 - t) ** 1.6                             # фаска ~45° з легким заокругленням
                c = np.clip(-d * sx + 0.5, 0, 1)
                gx, gy = r.normal(0, 0.0015, 2)
                tp = TOP + r.normal(0, 0.15) + gx * u + gy * v
                k_sw = int(r.integers(len(SWM)))
                attrs.append(dict(chip=r.uniform(0, 1) ** 2, sw=k_sw, sl=(slice(y0, y1), slice(x0, x1))))
                # --- взірець з фото: довга сторона плити = довга сторона взірця
                long_u = w >= h
                A = (u / w + 0.5) if long_u else (v / h + 0.5)
                B = (v / h + 0.5) if long_u else (u / w + 0.5)
                sw = SWM[k_sw]
                Hs, Ws = sw.shape[:2]
                target = max(w, h) / min(w, h)
                rs = Ws / Hs
                if rs < target / 1.4:        # квадратний взірець: беремо смугу і трохи розтягуємо вздовж
                    cw, ch = Ws, Ws / (target / 1.4)
                elif rs > target * 1.25:
                    cw, ch = Hs * target * 1.25, Hs
                else:
                    cw, ch = Ws, Hs
                ox, oy = r.uniform(0, Ws - cw), r.uniform(0, Hs - ch)
                if r.random() < 0.5:
                    A = 1 - A
                if r.random() < 0.5:
                    B = 1 - B
                mx = (ox + np.clip(A, 0, 1) * (cw - 1)).astype(np.float32)
                my = (oy + np.clip(B, 0, 1) * (ch - 1)).astype(np.float32)
                col = cv2.remap(sw, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                ax_idx = int(round(((ang if long_u else ang + 90) % 180) / 45)) % 4
                idx = len(attrs) - 1
                sl = (slice(y0, y1), slice(x0, x1))
                upd = c > cov[sl]
                prof[sl] = np.where(upd, p, prof[sl])
                top[sl] = np.where(upd, tp, top[sl])
                sid[sl] = np.where(upd, idx, sid[sl])
                tmap[sl] = np.where(upd[..., None], col, tmap[sl])
                axis[sl] = np.where(upd, ax_idx, axis[sl])
                cov[sl] = np.maximum(cov[sl], c)

    mm = lambda v: v * ppm
    warp = pnoise((H, W), mm(45), rng)
    mid = pnoise((H, W), mm(12), rng)
    fine = pnoise((H, W), mm(0.9), rng)
    fine2 = pnoise((H, W), mm(0.6), rng)
    pores = np.clip(pnoise((H, W), mm(0.7), rng) - 2.6, 0, None)
    aggr = pnoise((H, W), mm(1.6), rng)

    stone_top = top + 0.12 * warp + 0.05 * mid + 0.04 * fine - 0.25 * pores
    stone_h = 1.8 + (stone_top - 1.8) * prof
    sand_h = SAND + 0.30 * fine2 + 0.25 * mid
    height = np.where(cov > 0, np.maximum(stone_h, sand_h * (1 - cov) + stone_h * cov), sand_h)
    height = np.maximum(height, sand_h)

    # --- колір (sRGB): розкладка плям і палітра з фото + волокна в повній роздільності
    def lut(f):
        return np.array([f(a) if a else 0 for a in attrs], np.float32)[sid]
    inside = (cov > 0.5).astype(np.float32)
    wsum = gaussian_filter(inside, mm(12), mode="wrap") + 1e-4
    t1 = tmap[..., 0]
    m1 = gaussian_filter(t1 * inside, mm(12), mode="wrap") / wsum
    sd = np.sqrt(np.maximum(gaussian_filter(t1 ** 2 * inside, mm(12), mode="wrap") / wsum - m1 ** 2, 0))

    def aniso(angle, along, across):
        """Видовжений шум: волокна смуг уздовж довгої сторони плити."""
        fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.rfftfreq(W)[None, :]
        t = math.radians(angle)
        fu = fx * math.cos(t) + fy * math.sin(t)     # частота вздовж напрямку (cos t, sin t), Y — вниз
        fv = -fx * math.sin(t) + fy * math.cos(t)
        filt = np.exp(-(fu * along) ** 2 - (fv * across) ** 2)
        o = np.fft.irfft2(np.fft.rfft2(rng.standard_normal((H, W))) * filt, s=(H, W))
        return ((o - o.mean()) / (o.std() + 1e-9)).astype(np.float32)
    # широкі смуги вздовж довгої сторони плити (злегка хвилясті)
    fib = np.stack([aniso(a_, mm(45), mm(6.0)) for a_ in (0, 45, 90, 135)])
    S_ = np.take_along_axis(fib, axis[None].astype(np.int64), 0)[0]
    gx_, gy_ = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    dx_ = pnoise((H, W), mm(70), rng) * mm(2.5)
    dy_ = pnoise((H, W), mm(70), rng) * mm(2.5)
    S_ = cv2.remap(S_, gx_ + dx_, gy_ + dy_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    # дрібне зерно: перехід «персиковий ↔ сірий» складається з окремих крапинок, як на фото
    G_ = 0.7 * pnoise((H, W), mm(1.1), rng) + 0.3 * pnoise((H, W), mm(2.6), rng)
    z = (t1 - m1) / (sd + 6) + 0.55 * S_ * (0.45 + np.clip(sd / 25, 0, 1.2)) + 0.95 * G_
    t1d = m1 + (sd + 6) * np.tanh(2.6 * z) / np.tanh(2.6)
    # підгонка під камінь з фото: середнє і розкид (пропорції сірого й персикового),
    # без проміжних «лілових» тонів — на фото перехід складається з окремих зерен
    for k, at in enumerate(attrs):
        if not at:
            continue
        sl = at["sl"]
        msk = (sid[sl] == k) & (cov[sl] > 0.5)
        vals = t1d[sl][msk]
        if vals.size < 50:
            continue
        q = SWQ[at["sw"]]
        mu_s = float(np.mean(q)); sd_s = float(np.std(q)) * 1.5 + 4
        sub = t1d[sl]
        sub[msk] = np.clip(mu_s + (vals - vals.mean()) * sd_s / (vals.std() + 1e-6), q[3], q[-4])
        t1d[sl] = sub
    tt = np.stack([t1d, 0.9 * tmap[..., 1], 0.6 * tmap[..., 2]], -1)   # трохи приглушуємо кольоровий шум фото (зелені відблиски)
    stone_rgb = MU[None, None, :] + tt @ E
    lum = stone_rgb.mean(-1)
    # калібрування за фото: середні кольори персикової, нейтральної та сірої зон
    # (лінійна підгонка по кожному каналу, найменші квадрати з вагою = частці зони)
    sp = stone_rgb[cov > 0.9]
    warm = sp[:, 0] - sp[:, 2]
    zs = [warm > 28, (warm >= 10) & (warm <= 28), warm < 10]
    src = np.stack([sp[z].mean(0) for z in zs])
    dst = np.array([ZONES["peach_rgb"], ZONES["mid_rgb"], ZONES["grey_rgb"]], np.float32)
    wts = np.sqrt(np.array([z.mean() for z in zs]) + 0.05)
    for ch in range(3):
        A_ = np.c_[src[:, ch], np.ones(3)] * wts[:, None]
        a_, b_ = np.linalg.lstsq(A_, dst[:, ch] * wts, rcond=None)[0]
        stone_rgb[..., ch] = stone_rgb[..., ch] * a_ + b_
    speck = pnoise((H, W), mm(0.8), rng)
    stone_rgb += (8 * fine)[..., None]
    stone_rgb += (np.where(speck > 1.8, 32, 0) + np.where(speck < -2.0, -26, 0))[..., None]
    edge = (1 - prof) * (cov > 0)
    stone_rgb += (edge * 8 * lut(lambda a: a["chip"]))[..., None]
    stone_rgb -= (20 * pores)[..., None]
    sand_rgb = np.array([92, 86, 82], np.float32)[None, None, :] * (1 + 0.10 * fine2)[..., None]
    sand_rgb += (np.where(fine2 > 1.7, 60, 0))[..., None]
    a = cov[..., None]
    rgb = stone_rgb * a + sand_rgb * (1 - a)
    pm = np.zeros_like(lum)

    # --- AO
    hb1 = gaussian_filter(height, sigma=mm(2.5), mode="wrap")
    hb2 = gaussian_filter(height, sigma=mm(8), mode="wrap")
    cav = np.maximum(hb1 - height, 0) * 0.18 + np.maximum(hb2 - height, 0) * 0.10
    ao = np.clip(1 - cav, 0.35, 1)
    rgb *= (0.75 + 0.25 * ao)[..., None]

    # --- roughness
    rough = (0.76 + 0.04 * mid + 0.03 * fine - 0.04 * pm) * cov + (0.95 + 0.02 * fine2) * (1 - cov)
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
    Image.fromarray(u8(disp * 255)).save(f"{pre}_Displacement8.jpg", quality=95)  # для Lumion/SketchUp
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
