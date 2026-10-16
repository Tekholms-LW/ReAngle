"""Depth-based forward warping (point splat with z-buffer). Pure numpy, CPU-friendly."""
from __future__ import annotations

import numpy as np


def depth_from_relative(rel: np.ndarray, near: float = 1.0, far: float = 10.0, inverse: bool = True) -> np.ndarray:
    """Map a normalized depth map in [0,1] to metric-ish depth.

    Video Depth Anything style outputs are relative *inverse* depth (1 = near).
    """
    r = np.clip(rel.astype(np.float64), 0.0, 1.0)
    if inverse:
        inv = 1.0 / far + r * (1.0 / near - 1.0 / far)
        return 1.0 / inv
    return near + r * (far - near)


def warp_frame(img: np.ndarray, depth: np.ndarray, K: np.ndarray, T: np.ndarray):
    """Warp one frame.

    img: (H,W,C) float, depth: (H,W) >0, K: 3x3, T: 4x4 source->target.
    Returns warped (H,W,C), hole_mask (H,W) float {0,1} (1 = no source pixel), warped_depth (H,W) (0 in holes).
    """
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.reshape(-1)
    valid = np.isfinite(z) & (z > 0)
    pix = np.stack([u.reshape(-1), v.reshape(-1), np.ones(H * W)], 0)
    pts = (np.linalg.inv(K) @ pix) * z                       # 3xN in source cam
    pts = T[:3, :3] @ pts + T[:3, 3:4]                       # into target cam
    zt = pts[2]
    valid &= zt > 1e-6
    proj = K @ pts
    with np.errstate(divide="ignore", invalid="ignore"):
        ut = np.rint(proj[0] / zt)
        vt = np.rint(proj[1] / zt)
    valid &= (ut >= 0) & (ut < W) & (vt >= 0) & (vt < H)
    idx = np.nonzero(valid)[0]
    # Draw far-to-near so nearer points overwrite (z-buffer).
    idx = idx[np.argsort(-zt[idx], kind="stable")]
    ui, vi = ut[idx].astype(np.int64), vt[idx].astype(np.int64)
    C = img.shape[2]
    warped = np.zeros((H, W, C), dtype=img.dtype)
    wdepth = np.zeros((H, W), dtype=np.float64)
    hit = np.zeros((H, W), dtype=bool)
    flat = img.reshape(-1, C)
    warped[vi, ui] = flat[idx]
    wdepth[vi, ui] = zt[idx]
    hit[vi, ui] = True
    return warped, (~hit).astype(np.float32), wdepth


def fill_cracks(img: np.ndarray, hole: np.ndarray, depth: np.ndarray, passes: int = 2):
    """Fill thin splat cracks: a hole pixel with >= 2 valid 4-neighbours takes their mean.

    Large disocclusions (whole regions) stay holes so the video model can inpaint them.
    Returns (img, hole, depth) copies.
    """
    img, hole, depth = img.copy(), hole.astype(bool).copy(), depth.copy()
    for _ in range(max(0, passes)):
        valid = ~hole
        acc = np.zeros_like(img, dtype=np.float64)
        dacc = np.zeros(hole.shape, dtype=np.float64)
        cnt = np.zeros(hole.shape, dtype=np.int32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            v = np.zeros_like(valid)
            sl_dst = (slice(max(dy, 0), hole.shape[0] + min(dy, 0)), slice(max(dx, 0), hole.shape[1] + min(dx, 0)))
            sl_src = (slice(max(-dy, 0), hole.shape[0] + min(-dy, 0)), slice(max(-dx, 0), hole.shape[1] + min(-dx, 0)))
            v[sl_dst] = valid[sl_src]
            a = np.zeros_like(acc); a[sl_dst] = img[sl_src]
            d = np.zeros_like(dacc); d[sl_dst] = depth[sl_src]
            acc += a * v[..., None]; dacc += d * v; cnt += v
        fill = hole & (cnt >= 2)
        if not fill.any():
            break
        img[fill] = (acc[fill] / cnt[fill][:, None]).astype(img.dtype)
        depth[fill] = dacc[fill] / cnt[fill]
        hole[fill] = False
    return img, hole.astype(np.float32), depth


def warp_video(frames: np.ndarray, depths: np.ndarray, K: np.ndarray, path: np.ndarray, crack_passes: int = 0):
    """frames (N,H,W,C), depths (N,H,W), path (N,4,4) -> (warped, masks, wdepths)."""
    if not (len(frames) == len(depths)):
        raise ValueError("frames/depths length mismatch")
    n = len(frames)
    if len(path) != n:  # resample path to frame count
        idx = np.round(np.linspace(0, len(path) - 1, n)).astype(int)
        path = path[idx]
    outs = [warp_frame(frames[i], depths[i], K, path[i]) for i in range(n)]
    if crack_passes > 0:
        outs = [fill_cracks(w, m, d, crack_passes) for w, m, d in outs]
    return (np.stack([o[0] for o in outs]), np.stack([o[1] for o in outs]), np.stack([o[2] for o in outs]))


# ---- disocclusion / out-of-frame fill ------------------------------------------------------------------------
def _shift(a: np.ndarray, dy: int, dx: int, fill=0):
    out = np.full_like(a, fill)
    H, W = a.shape[:2]
    out[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = a[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return out


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    out = m.copy()
    for _ in range(r):
        out = out | _shift(out, 0, 1, False) | _shift(out, 0, -1, False) | _shift(out, 1, 0, False) | _shift(out, -1, 0, False)
    return out


def _box(a: np.ndarray, r: int) -> np.ndarray:
    """Separable box blur (edge-clamped)."""
    if r <= 0:
        return a
    k = 2 * r + 1
    p = np.pad(a, [(r, r), (r, r)] + [(0, 0)] * (a.ndim - 2), mode="edge").astype(np.float64)
    c = np.cumsum(p, 0); c = np.concatenate([np.zeros_like(c[:1]), c], 0); p = (c[k:] - c[:-k]) / k
    c = np.cumsum(p, 1); c = np.concatenate([np.zeros_like(c[:, :1]), c], 1)
    return (c[:, k:] - c[:, :-k]) / k


def remove_see_through(img, hole, depth, ratio: float = 0.8, min_near: int = 5):
    """Far splats that leak through gaps in a nearer surface (bright specks in hair): a valid pixel whose 8-neighbours
    are mostly much nearer takes their mean colour and depth."""
    valid = hole < 0.5
    near_cnt = np.zeros(depth.shape, np.int32)
    acc = np.zeros(img.shape, np.float64)
    dacc = np.zeros(depth.shape, np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            d, v = _shift(depth, dy, dx), _shift(valid, dy, dx, False)
            near = v & (d < depth * ratio)
            near_cnt += near
            acc += _shift(img, dy, dx) * near[..., None]
            dacc += d * near
    fix = valid & (near_cnt >= min_near)
    img, depth = img.copy(), depth.copy()
    img[fix] = (acc[fix] / near_cnt[fix][:, None]).astype(img.dtype)
    depth[fix] = dacc[fix] / near_cnt[fix]
    return img, depth, int(fix.sum())


def _fill_runs_1d(img, hole, depth, axis: int):
    """Fill hole runs along rows (axis=1) or columns (axis=0) from the FARTHER end (background side, never the
    foreground); runs touching the frame border edge-extend from the only side they have."""
    if axis == 0:
        a, b, c, bd = _fill_runs_1d(img.transpose(1, 0, 2), hole.T, depth.T, 1)
        return a.transpose(1, 0, 2), b.T, c.T, bd.T
    img, hole, depth = img.copy(), hole.copy(), depth.copy()
    border = np.zeros(hole.shape, bool)
    orig = img.copy(); valid0 = ~hole
    H, W = hole.shape
    for y in np.nonzero(hole.any(1))[0]:
        row = hole[y]
        if row.all():
            continue
        e = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        for s, t in zip(np.nonzero(e == 1)[0], np.nonzero(e == -1)[0]):
            L = s - 1 if s > 0 else None
            R = t if t < W else None
            if L is None or R is None:
                # frame-border run: mirror the texture next to it (reflect about the last valid pixel) instead of
                # smearing one edge pixel across the gap; invalid mirror pixels fall back to the edge colour
                src = L if L is not None else R
                i = np.arange(t - s)
                # reflect about the last valid pixel; past the run's midpoint flip back (triangle wave) so a wide
                # gap never samples far into the frame, where it can pick up a bright/dark patch -> band
                half = max(1, (t - s + 1) // 2)
                d = np.abs(s + i - src)
                d = np.where(d > half, 2 * half - d, d)
                j = np.clip(src + np.sign(src - (s + i)) * d, 0, W - 1)
                # only mirror BACKGROUND: a mirrored subject (hair/coat near the edge) reads as a second figure.
                # Foreground samples fall back to the mean colour of the row's far pixels.
                rv = valid0[y] & (depth[y] > 0)
                far_thr = 0.8 * depth[y][rv].max() if rv.any() else 0
                far = rv & (depth[y] >= far_thr)
                bg = orig[y][far].mean(0) if far.any() else img[y, src]
                ok = valid0[y, j] & (depth[y, j] >= far_thr)
                img[y, s:t] = np.where(ok[:, None], orig[y, j], bg)
                depth[y, s:t] = depth[y, src]
                hole[y, s:t] = False
                border[y, s:t] = True
                continue
            src = L if depth[y, L] >= depth[y, R] else R
            # average a few pixels beyond the source end that are at least as far (pure background)
            sl = slice(max(0, src - 3), src + 1) if src == L else slice(src, min(W, src + 4))
            seg_d = depth[y, sl]
            keep = (seg_d >= depth[y, src] * 0.95) & ~hole[y, sl]
            colour = img[y, sl][keep].mean(0) if keep.any() else img[y, src]
            img[y, s:t] = colour
            depth[y, s:t] = depth[y, src]
            hole[y, s:t] = False
    return img, hole, depth, border


def prefill_holes(img, hole, depth, edge_dilate: int = 3, blur: int = 4):
    """Make a guide frame with no black: returns (filled_img, regen_mask) where regen_mask (H,W) in [0,1] marks
    pixels the video model should regenerate (holes dilated by `edge_dilate` px, feathered).

    1. remove far specks seen through near surfaces
    2. grow holes by edge_dilate px into the NEARER side of each depth edge (stretched / haloed silhouettes)
    3. fill each hole run from its farther neighbour (rows, then columns for top/bottom wedges)
    4. blur only inside the filled area so the fill has no streaks
    """
    img, depth, _ = remove_see_through(img, hole, depth)
    h = hole > 0.5
    grown = _dilate(h, edge_dilate)
    if grown.any() and (~h).any():
        # only eat into pixels that sit on a depth edge (foreground boundary), not into the flat background
        dmin = depth.copy(); dmax = depth.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sd = _shift(np.where(h, np.nan, depth), dy, dx, np.nan)
            with np.errstate(invalid="ignore"):
                dmin = np.fmin(dmin, sd); dmax = np.fmax(dmax, sd)
        edge = _dilate((dmax - dmin) > 0.15 * dmin, edge_dilate)
        h2 = h | (grown & edge)
    else:
        h2 = h
    out, rem, d, b1 = _fill_runs_1d(img, h2, np.where(h2, 0, depth), 1)
    out, rem, d, b2 = _fill_runs_1d(out, rem, d, 0)
    if rem.any():                                   # fully empty frame (degenerate): mid grey
        out[rem] = 0.5
    bord = b1 | b2
    sm = _box(out, blur)
    smb = _box(out, blur * 2)                       # border mirror: softer, so it reads as defocused background
    out = np.where(bord[..., None], smb, np.where(h2[..., None], sm, out)).astype(img.dtype)
    regen = np.clip(_box(_dilate(h2, 2).astype(np.float64), 2), 0, 1)
    if bord.any():                                  # feather border regions wide so the model blends, not invents
        regen = np.maximum(regen, np.clip(_box(_dilate(bord, 6).astype(np.float64), 12) * 1.5, 0, 1))
    regen = regen.astype(np.float32)
    return out, regen
