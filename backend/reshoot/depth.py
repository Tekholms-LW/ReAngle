"""First-frame depth preview for the 3D camera-path editor.

Returns a small RGB + relative-inverse-depth grid; the browser turns it into a point cloud with the same
intrinsics/depth mapping DepthReproject uses (hfov 60, near 1, far 10), so the editor preview matches the warp.
Providers: ComfyUI Video Depth Anything (local backend) with a heuristic fallback (no GPU / mock mode).
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import uuid
from pathlib import Path

import numpy as np

from .backends import SAFE

PARAMS = {"hfov_deg": 60.0, "near": 1.0, "far": 10.0, "pivot_depth": 3.0}


def first_frame_rgb(src: Path, w: int, h: int) -> np.ndarray:
    vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vf", vf, "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def heuristic_depth(rgb: np.ndarray) -> np.ndarray:
    """Ground-plane guess: bottom of frame near, top far, nudged by brightness. Values: rel. inverse depth 0..1."""
    h, w, _ = rgb.shape
    y = np.linspace(0.0, 1.0, h)[:, None]
    lum = rgb.astype(np.float32).mean(axis=2) / 255.0
    return np.clip(0.1 + 0.75 * y**1.5 + 0.15 * (lum - lum.mean()), 0, 1).astype(np.float32)


def _png_bytes(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    return subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                           "-i", "-", "-f", "image2", "-c:v", "png", "-"], input=rgb.tobytes(),
                          capture_output=True, check=True).stdout


def comfy_depth(rgb: np.ndarray, base_url: str) -> np.ndarray:
    """Run Video Depth Anything (small) on one frame through ComfyUI's native API."""
    from .backends import NativeComfyBackend
    b = NativeComfyBackend(base_url)
    q = json.loads(b._http("GET", "/queue"))
    if q.get("queue_running") or q.get("queue_pending"):
        raise RuntimeError("ComfyUI busy")
    tmp = Path(os.environ.get("TEMP", "/tmp")) / f"reangle_f0_{uuid.uuid4().hex[:6]}.png"
    tmp.write_bytes(_png_bytes(rgb))
    name = b._upload(tmp)
    tmp.unlink(missing_ok=True)
    g = {"1": {"class_type": "LoadImage", "inputs": {"image": name}},
         "2": {"class_type": "LoadVideoDepthAnythingModel", "inputs": {"model": "video_depth_anything_vits.pth"}},
         "3": {"class_type": "VideoDepthAnythingProcess", "inputs": {"vda_model": ["2", 0], "images": ["1", 0],
               "input_size": 518, "max_res": 640, "precision": "fp16"}},
         "4": {"class_type": "VideoDepthAnythingOutput", "inputs": {"depths": ["3", 0], "colormap": "gray"}},
         "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": "reangle/depthprev"}}}
    import time
    import urllib.parse
    pid = json.loads(b._http("POST", "/prompt", json.dumps({"prompt": g}).encode(),
                             {"Content-Type": "application/json"}))["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < 120:
        time.sleep(1)
        h = json.loads(b._http("GET", f"/history/{pid}"))
        if pid in h and h[pid].get("status", {}).get("status_str") in ("success", "error"):
            break
    else:
        raise TimeoutError("depth preview timed out")
    if h[pid]["status"]["status_str"] != "success":
        raise RuntimeError("depth preview failed")
    f = h[pid]["outputs"]["5"]["images"][0]
    png = b._http("GET", "/view?" + urllib.parse.urlencode({"filename": f["filename"], "subfolder": f["subfolder"],
                                                           "type": f["type"]}))
    hh, ww, _ = rgb.shape
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-vf", f"scale={ww}:{hh}", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], input=png, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(hh, ww).astype(np.float32) / 255.0


def depth_preview(src: Path, cache: Path, provider: str, base_url: str | None = None, grid_w: int = 160) -> dict:
    if cache.is_file():
        return json.loads(cache.read_text())
    W, H = SAFE["width"], SAFE["height"]
    try:  # match the clip's aspect (square/portrait) so the 3D preview never crops the subject
        from .backends import probe
        from .hardware import aspect_of, res_for_aspect
        info = probe(src)
        W, H = (int(x) for x in res_for_aspect(f"{W}x{H}", aspect_of(info["width"], info["height"])).split("x"))
    except Exception:  # noqa: BLE001
        pass
    rgb = first_frame_rgb(src, W, H)
    source = "heuristic"
    rel = None
    if provider == "comfy" and base_url:
        try:
            rel, source = comfy_depth(rgb, base_url), "video-depth-anything"
        except Exception as e:  # noqa: BLE001
            source = f"heuristic (depth model unavailable: {e})"
    if rel is None:
        rel = heuristic_depth(rgb)
    gw = grid_w
    gh = max(2, round(gw * H / W))
    ys = (np.arange(gh) + 0.5) * H / gh
    xs = (np.arange(gw) + 0.5) * W / gw
    yi, xi = ys.astype(int), xs.astype(int)
    rgb_s = rgb[yi][:, xi]
    rel_s = rel[yi][:, xi]
    out = {"width": W, "height": H, "grid_w": gw, "grid_h": gh, "source": source, **PARAMS,
           "rgb": base64.b64encode(np.ascontiguousarray(rgb_s).tobytes()).decode(),
           "depth": base64.b64encode((np.clip(rel_s, 0, 1) * 255).astype(np.uint8).tobytes()).decode()}
    if not source.startswith("heuristic ("):
        cache.write_text(json.dumps(out))
    return out
