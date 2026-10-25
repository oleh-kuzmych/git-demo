"""Вирізає реальні камені з фото колорміксу для текстур.

1. Знаходить шви (темні тонкі лінії) і дві точки сходження.
2. Вирівнює перспективу (метрична ректифікація з двох ортогональних напрямків).
3. Виділяє окремі камені, відкидає ті, що зачепили шов, траву чи край кадру.
4. Розвертає кожен камінь довгою стороною по X (смуги колорміксу йдуть уздовж
   довгої сторони) і вирівнює яскравість між фото.

Результат: reference/swatches/*.png + swatches.json
Запуск: python3 extract_swatches.py reference/colormix_2.jpg reference/colormix_3.jpg
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "reference", "swatches")


def joints_mask(gray, k=7):
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    _, bw = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def vanishing_point(seg, rng):
    hl = np.cross(np.c_[seg[:, :2], np.ones(len(seg))], np.c_[seg[:, 2:], np.ones(len(seg))])
    hl /= np.linalg.norm(hl[:, :2], axis=1, keepdims=True)
    mid = (seg[:, :2] + seg[:, 2:]) / 2
    d1 = seg[:, 2:] - seg[:, :2]
    n1 = np.linalg.norm(d1, axis=1)
    best = None
    for _ in range(3000):
        i, j = rng.choice(len(hl), 2, replace=False)
        v = np.cross(hl[i], hl[j])
        if abs(v[2]) < 1e-9:
            continue
        v = v / v[2]
        d2 = v[:2] - mid
        cos = np.abs((d1 * d2).sum(1)) / (n1 * np.linalg.norm(d2, axis=1) + 1e-9)
        inl = cos > np.cos(np.radians(1.5))
        score = (n1 * inl).sum()
        if best is None or score > best[0]:
            best = (score, inl)
    _, _, vt = np.linalg.svd(hl[best[1]])
    return vt[-1] / vt[-1][2]


def rectify(img):
    H0, W0 = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    seg = cv2.HoughLinesP(joints_mask(g), 1, np.pi / 360, 40, minLineLength=40, maxLineGap=6).reshape(-1, 4).astype(float)
    ang = np.degrees(np.arctan2(seg[:, 3] - seg[:, 1], seg[:, 2] - seg[:, 0])) % 180
    rng = np.random.default_rng(0)
    vA = vanishing_point(seg[ang < 75], rng)
    vB = vanishing_point(seg[ang > 105], rng)
    c = np.array([W0 / 2, H0 / 2])
    f2 = -np.dot(vA[:2] - c, vB[:2] - c)
    if f2 <= 0:
        raise RuntimeError("точки сходження неортогональні — фото не підходить")
    f = np.sqrt(f2)
    K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]])
    Ki = np.linalg.inv(K)
    r1 = Ki @ vA; r1 /= np.linalg.norm(r1)
    r2 = Ki @ vB; r2 -= r1 * (r1 @ r2); r2 /= np.linalg.norm(r2)
    Hm = K @ np.c_[r1, r2, Ki @ np.array([W0 / 2, H0 * 0.8, 1.0])]
    Hi = np.linalg.inv(Hm)
    corn = np.array([[0, H0 * 0.25], [W0, H0 * 0.25], [W0, H0], [0, H0]], float)
    wc = cv2.perspectiveTransform(corn[None], Hi)[0]
    (x0, y0), (x1, y1) = wc.min(0), wc.max(0)
    s = 900 / max(x1 - x0, y1 - y0)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s], [0, 0, 1]]) @ Hi
    size = (int((x1 - x0) * s), int((y1 - y0) * s))
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0)), M


def has_inner_joint(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    raw = (bh > 12).astype(np.uint8)
    hz = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((1, max(5, int(w * 0.85))), np.uint8))
    vt = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((max(5, int(h * 0.85)), 1), np.uint8))
    return hz.any() or vt.any()


def streak_angle(crop):
    """Напрям смуг (структурний тензор): 0° — уздовж X, 90° — уздовж Y; та виразність."""
    g = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    jxx, jyy, jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
    grad = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))   # напрям градієнта
    coh = np.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + 1e-9)
    return (grad + 90) % 180, coh


def stones(rect, M):
    g = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    valid = (g > 20) & ~((hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 60))
    valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    raw = (bh > max(7, np.percentile(bh[valid], 72))).astype(np.uint8)
    est = np.median([d for d in (np.diff(np.where(raw.sum(0) > raw.shape[0] * 0.1)[0])) if d > 5] or [60])
    Lk = int(np.clip(est * 0.4, 15, 35))
    hz = cv2.morphologyEx(cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((1, Lk), np.uint8)), cv2.MORPH_CLOSE, np.ones((1, 15), np.uint8))
    vt = cv2.morphologyEx(cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((Lk, 1), np.uint8)), cv2.MORPH_CLOSE, np.ones((15, 1), np.uint8))
    joint = cv2.dilate(hz | vt, np.ones((3, 3), np.uint8))
    k, _, st, _ = cv2.connectedComponentsWithStats(((joint == 0) & valid).astype(np.uint8), 4)
    out = []
    for i in range(1, k):
        x, y, w, h, a = st[i]
        if a < 900 or w < 28 or h < 28 or a / (w * h) < 0.8:
            continue
        m = int(round(min(w, h) * 0.10)) + 2
        crop = rect[y + m:y + h - m, x + m:x + w - m]
        if crop.size == 0 or (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 25).any() or has_inner_joint(crop):
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        if gray.std() < 11:          # «пласкі» взірці без малюнка (розмиті на дальньому плані)
            continue
        out.append((sharp, crop))
    if not out:
        return []
    cut = np.percentile([s for s, _ in out], 30)   # чверть найрозмитіших (дальній план) відкидаємо
    return [c for s, c in out if s >= cut]


def main(paths):
    os.makedirs(OUT, exist_ok=True)
    allc, meta = [], []
    for p in paths:
        rect, M = rectify(cv2.imread(p))
        cs = stones(rect, M)
        print(os.path.basename(p), "каменів:", len(cs))
        allc.append(cs)
    # вирівнювання яскравості між фото (різна експозиція)
    means = [np.mean([c.reshape(-1, 3).mean(0) for c in cs], 0) for cs in allc]
    target = np.mean(means, 0)
    for fi, (p, cs) in enumerate(zip(paths, allc)):
        gain = target / means[fi]
        for j, c in enumerate(cs):
            c = np.clip(c.astype(np.float32) * gain[None, None, :], 0, 255).astype(np.uint8)
            ang, coh = streak_angle(c)
            vertical = 45 < ang < 135
            # смуги колорміксу кладемо вздовж довгої сторони плити (вісь X взірця);
            # якщо смуг майже не видно — орієнтуємо за формою каменю
            if (coh > 0.12 and vertical) or (coh <= 0.12 and c.shape[0] > c.shape[1] * 1.15):
                c = cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE)
            name = f"{os.path.splitext(os.path.basename(p))[0]}_{j:02d}.png"
            cv2.imwrite(os.path.join(OUT, name), c)
            meta.append({"file": name, "w": int(c.shape[1]), "h": int(c.shape[0])})
    # еталонні кольори зон з фото (для калібрування текстур): персикові й сірі ділянки
    px = np.concatenate([cv2.imread(os.path.join(OUT, e["file"]))[..., ::-1].reshape(-1, 3) for e in meta]).astype(float)
    warm = px[:, 0] - px[:, 2]
    zones = {"peach_rgb": px[warm > 28].mean(0).round(1).tolist(), "grey_rgb": px[warm < 10].mean(0).round(1).tolist(),
             "mid_rgb": px[(warm >= 10) & (warm <= 28)].mean(0).round(1).tolist(),
             "peach_share": round(float((warm > 28).mean()), 3), "grey_share": round(float((warm < 10).mean()), 3)}
    print("еталон:", zones)
    json.dump({"swatches": meta, "mean_bgr": target.round(1).tolist(), "zones": zones}, open(os.path.join(OUT, "swatches.json"), "w"), indent=1)
    print("усього взірців:", len(meta))


if __name__ == "__main__":
    main(sys.argv[1:] or [os.path.join(ROOT, "reference", f) for f in ("colormix_2.jpg", "colormix_3.jpg")])
