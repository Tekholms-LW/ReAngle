"""Long clips + tiered quality on a local/remote ComfyUI: one depth pass + one global camera path for the whole
clip, warped guide precomputed here, then LTX renders overlapping chunks. Each chunk after the first is
conditioned on the last frames of the previous chunk's output; overlaps are crossfaded; original audio muxed.

Resumable: every artefact lives in a per-render work dir; finished chunks are skipped on retry.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import camerapath  # noqa: F401  (puts the node pack on sys.path)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workflows"))


class Cancelled(Exception):
    pass


@dataclass
class Plan:
    total: int                  # frames in the re-shoot
    window: int                 # frames per chunk (8n+1)
    overlap: int                # frames conditioned on the previous chunk (8n+1)
    starts: list[int] = field(default_factory=list)

    def chunks(self) -> list[tuple[int, int]]:
        return [(s, min(s + self.window, self.total)) for s in self.starts]


def round8(n: int) -> int:
    return max(9, (n - 1) // 8 * 8 + 1)


def plan_chunks(total: int, window: int, overlap: int) -> Plan:
    """Overlapping windows covering [0,total). Last window is shifted back so every chunk has `window` frames."""
    window, overlap = round8(window), round8(overlap) if overlap >= 9 else overlap
    if total <= window:
        return Plan(round8(total), round8(total), 0, [0])
    stride = window - overlap
    starts = list(range(0, total - window, stride))
    starts.append(total - window)
    starts = sorted(set(starts))
    # every start (after the first) must be inside the previous chunk by at least `overlap` frames
    for a, b in zip(starts, starts[1:]):
        assert b - a <= stride and b > a
    return Plan(total, window, overlap, starts)


def blend_into(acc: np.ndarray, filled: int, chunk: np.ndarray, start: int) -> int:
    """Crossfade `chunk` (frames start..start+len) into acc where acc already holds [0,filled)."""
    end = start + len(chunk)
    ov = max(0, filled - start)
    if ov:
        w = (np.arange(1, ov + 1, dtype=np.float32) / (ov + 1))[:, None, None, None]
        acc[start:filled] = acc[start:filled] * (1 - w) + chunk[:ov] * w
    acc[start + ov:end] = chunk[ov:]
    return max(filled, end)


# ---------- video helpers (ffmpeg pipes, no extra deps)
def read_frames(path: Path, w: int, h: int, gray: bool = False) -> np.ndarray:
    pf, c = ("gray", 1) if gray else ("rgb24", 3)
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                        "-pix_fmt", pf, "-"], capture_output=True, check=True)
    a = np.frombuffer(r.stdout, np.uint8)
    return a.reshape(-1, h, w, c) if not gray else a.reshape(-1, h, w)


def write_frames(path: Path, frames: np.ndarray, fps: float, audio_from: Path | None = None, crf: int = 14) -> Path:
    n, h, w = frames.shape[:3]
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}",
           "-i", "-"]
    if audio_from:
        cmd += ["-i", str(audio_from), "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames, dtype=np.uint8).tobytes(), check=True, capture_output=True)
    return path


# ---------- camera + warp for the whole clip
def global_poses(preset: str, angle_deg: float, path: str | None, n: int) -> tuple[np.ndarray, bool]:
    if path:
        from reshoot_nodes.camera_json import path_from_json
        p = path_from_json(path, n)
        return p["poses"], bool(p["freeze_time"])
    from reshoot_nodes.camera import camera_path
    return camera_path(preset, angle_deg, n, 3.0), False


def warp_guide(frames: np.ndarray, depth_u8: np.ndarray, poses: np.ndarray, freeze: bool, fill: bool = False):
    """Warped guide (N,H,W,3) uint8. fill=True: also returns a regen mask (N,H,W) uint8 (255 = let the model
    generate) and the guide has its holes pre-filled from the background side instead of black."""
    from reshoot_nodes.camera import intrinsics
    from reshoot_nodes.reproject import depth_from_relative, prefill_holes, warp_video
    imgs = frames.astype(np.float32) / 255.0
    metric = depth_from_relative(depth_u8.astype(np.float32) / 255.0, 1.0, 10.0, True)
    if freeze:
        imgs, metric = np.repeat(imgs[:1], len(poses), 0), np.repeat(metric[:1], len(poses), 0)
    n, h, w, _ = imgs.shape
    warped, holes, wdep = warp_video(imgs, metric, intrinsics(w, h, 60.0), poses, crack_passes=2)
    if not fill:
        return (np.clip(warped, 0, 1) * 255).astype(np.uint8)
    out = np.empty_like(warped); masks = np.empty(holes.shape, np.uint8)
    for i in range(n):              # same deterministic rule every frame -> consistent across chunks
        out[i], m = prefill_holes(warped[i], holes[i], wdep[i])
        masks[i] = (m * 255).astype(np.uint8)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8), masks


def _lab(rgb):
    c = np.clip(rgb / 255.0, 0, 1)
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 0.2069, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.9505, 1.0, 1.089])
    c = xyz @ np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]).T
    c = np.clip(c, 0, 1)
    return np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c) * 255


def color_match_lab(frames: np.ndarray, ref: np.ndarray, known: np.ndarray | None) -> np.ndarray:
    """One LAB mean/std transform per chunk (stats over all its frames, only where the guide holds real warped
    pixels), so every chunk keeps the source's brightness, contrast and tint. Gains are clamped to stay subtle."""
    lo, lr = _lab(frames.astype(np.float64)), _lab(ref.astype(np.float64))
    sel = known if known is not None and known.sum() > 1000 else np.ones(frames.shape[:3], bool)
    out = lo.copy()
    for ch in range(3):
        a, b = lo[..., ch][sel], lr[..., ch][sel]
        gain = np.clip(b.std() / max(a.std(), 1e-6), 0.8, 1.25)
        out[..., ch] = (lo[..., ch] - a.mean()) * gain + b.mean()
    return _rgb(out).astype(np.float32)


@dataclass
class RenderOpts:
    width: int = 640
    height: int = 352
    fps: int = 24
    frames: int = 41
    window: int = 41
    overlap: int = 9
    upscale: bool = False
    unet: str | None = None
    fill: bool = True
    anchor: float = 0.85         # later chunks: I2V strength of the warped source frame at chunk start (0 = off)
    prev_strength: float = 0.6   # later chunks: strength of the previous chunk's tail guide
    color_match: bool = True     # match each chunk's LAB mean/std to the guide in known (non-filled) pixels


class ChunkedRenderer:
    """Drives a NativeComfyBackend-like client (needs: _http, _upload, run_prompt, free, preflight)."""

    def __init__(self, client, work: Path, opts: RenderOpts, progress: Callable, cancelled: Callable[[], bool]):
        self.c, self.work, self.o, self.progress, self.cancelled = client, work, opts, progress, cancelled
        work.mkdir(parents=True, exist_ok=True)
        self.stats: dict = {"chunks": [], "peak_vram_mb": 0}

    def _check(self):
        if self.cancelled():
            raise Cancelled("cancelled")

    def depth(self, clip: Path) -> np.ndarray:
        f = self.work / "depth.npy"
        if f.exists():
            return np.load(f)
        o = self.o
        dg = getattr(self.c, "depth_graph", None)
        if dg is None:
            try:
                from .backends import NativeComfyBackend
            except ImportError:
                from backends import NativeComfyBackend
            dg = lambda v, oo: NativeComfyBackend.depth_graph(None, v, oo)  # noqa: E731
        g, out_node = dg(self.c._upload(clip), o)
        out = self.c.run_prompt(g, out_node, kind="images", label="depth", expected_s=60)
        d = self.work / "depth"; d.mkdir(exist_ok=True)
        for i, data in enumerate(out):
            (d / f"{i:05d}.png").write_bytes(data)
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(d / "%05d.png"), "-vf", f"scale={o.width}:{o.height}",
                            "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True)
        dep = np.frombuffer(r.stdout, np.uint8).reshape(-1, o.height, o.width)
        np.save(f, dep)
        return dep

    def run(self, clip: Path, preset: str, angle_deg: float, path: str | None, out: Path) -> Path:
        from build_reshoot_workflow import build
        o = self.o
        src = read_frames(clip, o.width, o.height)[: o.frames]
        o.frames = len(src)
        plan = plan_chunks(o.frames, o.window, o.overlap)
        if plan.total != len(src):        # single chunk shorter than window: LTX needs 8n+1
            src = src[: plan.total]; o.frames = plan.total
        (self.work / "plan.json").write_text(json.dumps({"opts": o.__dict__, "starts": plan.starts,
                                                         "window": plan.window, "overlap": plan.overlap}))
        self.progress(0.02, "Estimating depth for the whole clip")
        self.c.free(); self._check()
        dep = self.depth(clip)[: o.frames]
        guide_f = self.work / ("guide_fill.mp4" if o.fill else "guide.mp4")
        mask_f = self.work / "regen_mask.mp4"
        if not guide_f.exists() or (o.fill and not mask_f.exists()):
            self.progress(0.05, "Warping the clip along your camera path" + (" and filling revealed areas" if o.fill else ""))
            poses, freeze = global_poses(preset, angle_deg, path, o.frames)
            wg = warp_guide(src, dep, poses, freeze, fill=o.fill)
            if o.fill:
                wg, m = wg
                write_frames(mask_f, np.repeat(m[..., None], 3, -1), o.fps, crf=4)
            write_frames(guide_f, wg, o.fps, crf=12)
        guide = read_frames(guide_f, o.width, o.height)
        regen = read_frames(mask_f, o.width, o.height) if o.fill else None
        ow, oh = (o.width * 2, o.height * 2) if o.upscale else (o.width, o.height)
        acc = np.zeros((o.frames, oh, ow, 3), np.float32)
        filled, n = 0, len(plan.starts)
        for k, (s, e) in enumerate(plan.chunks()):
            self._check()
            cf = self.work / f"chunk_{k:02d}.mp4"
            span = (0.08 + 0.9 * k / n, 0.08 + 0.9 * (k + 1) / n)
            label = f"Chunk {k + 1}/{n} (frames {s}-{e - 1})"
            if not cf.exists():
                prev = min(plan.overlap, filled - s) if k else 0
                prev = round8(prev) if prev >= 9 else (1 if prev else 0)
                g = build(upscale=o.upscale, width=o.width, height=o.height, frames=e - s, fps=o.fps,
                          unet=o.unet, chunk=True, prev_frames=prev, fill=o.fill,
                          anchor=o.anchor if o.fill else 0.0, prev_strength=o.prev_strength if o.fill else 1.0)
                files = {"src": write_frames(self.work / f"in_{k:02d}.mp4", src[s:e], o.fps),
                         "guide": write_frames(self.work / f"guide_{k:02d}.mp4", guide[s:e], o.fps, crf=12)}
                if o.fill:
                    files["mask"] = write_frames(self.work / f"mask_{k:02d}.mp4", regen[s:e], o.fps, crf=4)
                if prev:
                    tail = acc[s:s + prev]
                    if o.upscale:
                        tail = tail.reshape(prev, o.height, 2, o.width, 2, 3).mean(axis=(2, 4))
                    files["prev"] = write_frames(self.work / f"prev_{k:02d}.mp4", tail.clip(0, 255), o.fps, crf=12)
                g = self.c.patch_graph(g) if hasattr(self.c, "patch_graph") else g
                g["1"]["inputs"]["video"] = self.c._upload(files["src"])
                g["23"]["inputs"]["video"] = self.c._upload(files["guide"])
                if o.fill:
                    g["25"]["inputs"]["video"] = self.c._upload(files["mask"])
                if prev:
                    g["24"]["inputs"]["video"] = self.c._upload(files["prev"])
                self.c.free(); self._check()
                self.c.preflight()
                t0 = time.time()
                data = self.c.run_prompt(g, "90", kind="video", label=label, span=span)
                cf.write_bytes(data[0])
                self.stats["chunks"].append({"k": k, "frames": e - s, "seconds": round(time.time() - t0, 1),
                                             "peak_vram_mb": self.c.last_peak_mb})
                self.stats["peak_vram_mb"] = max(self.stats["peak_vram_mb"], self.c.last_peak_mb or 0)
            frames = read_frames(cf, ow, oh).astype(np.float32)[: e - s]
            if o.color_match and not o.upscale:
                known = (regen[s:e, ..., 0] < 25) if regen is not None else None
                frames = color_match_lab(frames, guide[s:e].astype(np.float32), known)
            filled = blend_into(acc, filled, frames, s)
            self.progress(span[1], f"{label} done")
        self.c.free()
        write_frames(out, acc.clip(0, 255), o.fps, audio_from=clip, crf=16)
        (self.work / "stats.json").write_text(json.dumps(self.stats, indent=1))
        return out
