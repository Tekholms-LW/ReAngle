"""Per-chunk drift of a re-shoot vs its warped guide, measured only where the guide is real (regen mask ~0).
usage: python drift_metrics.py out.mp4 guide.mp4 mask.mp4 [W H] [chunk_starts...]"""
import subprocess, sys
import numpy as np


def read(p, w, h, gray=False):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt",
                          "gray" if gray else "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape((-1, h, w) if gray else (-1, h, w, 3))


def lab(rgb):
    c = rgb.astype(np.float64) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def box(a, r=4):
    k = 2 * r + 1; p = np.pad(a, r, mode="edge")
    c = np.cumsum(np.cumsum(np.pad(p, ((1, 0), (1, 0))), 0), 1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / k / k


def ssim_map(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    mx, my = box(x), box(y); vx = box(x * x) - mx ** 2; vy = box(y * y) - my ** 2; cxy = box(x * y) - mx * my
    C1, C2 = 6.5, 58.5
    return ((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))


def main():
    out, guide, mask = sys.argv[1:4]
    W = int(sys.argv[4]) if len(sys.argv) > 4 else 448; H = int(sys.argv[5]) if len(sys.argv) > 5 else 448
    starts = [int(s) for s in sys.argv[6:]] or [0, 33, 57, 81]
    o, g, m = read(out, W, H), read(guide, W, H), read(mask, W, H, gray=True)
    n = min(len(o), len(g), len(m)); bounds = starts + [n]
    face = np.zeros((H, W), bool); face[int(.12 * H):int(.6 * H), int(.3 * W):int(.7 * W)] = True
    print("chunk frames   SSIM(known) SSIM(face) dL*   dStdL*  da*   db*")
    for k in range(len(starts)):
        s, e = bounds[k], bounds[k + 1]
        ss, sf, dl, ds, da, db = [], [], [], [], [], []
        for i in range(s, e):
            known = m[i] < 25
            lo, lg = lab(o[i]), lab(g[i])
            sm = ssim_map(lo[..., 0], lg[..., 0])
            ss.append(sm[known].mean()); sf.append(sm[known & face].mean())
            dl.append(lo[..., 0][known].mean() - lg[..., 0][known].mean())
            ds.append(lo[..., 0][known].std() - lg[..., 0][known].std())
            da.append(lo[..., 1][known].mean() - lg[..., 1][known].mean())
            db.append(lo[..., 2][known].mean() - lg[..., 2][known].mean())
        print(f"{k:5d} {s:3d}-{e - 1:<3d}   {np.mean(ss):.3f}      {np.mean(sf):.3f}     {np.mean(dl):+.2f} {np.mean(ds):+.2f}  "
              f"{np.mean(da):+.2f} {np.mean(db):+.2f}")


if __name__ == "__main__":
    main()
