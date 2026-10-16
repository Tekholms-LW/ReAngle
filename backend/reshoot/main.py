"""FastAPI app. Run: uvicorn reshoot.main:app --app-dir backend"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .backends import SAFE, make_backend, probe, safe_frames
from .jobs import JobQueue
from .presets import MAX_ANGLE_DEG, PRESETS, clamp_strength
from .camerapath import normalize_path, preset_to_json
from .depth import depth_preview
from . import hardware
import time as _time

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("RESHOOT_DATA", ROOT / "data"))
MAX_UPLOAD_MB = int(os.environ.get("RESHOOT_MAX_UPLOAD_MB", "200"))
MAX_INPUT_SECONDS = float(os.environ.get("RESHOOT_MAX_INPUT_SECONDS", "120"))
MAX_INPUT_PIXELS = 3840 * 2160
ALLOWED = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}

app = FastAPI(title="ReAngle", version="0.1.0")
queue = JobQueue(DATA, make_backend())


class AngleReq(BaseModel):
    preset: str                       # a preset id, or "custom" with `path`
    angle_deg: float = 15.0
    path: dict | None = None          # camera path JSON from the 3D editor


class JobReq(BaseModel):
    upload_id: str
    angles: list[AngleReq] = Field(min_length=1, max_length=8)
    duration_s: float | None = None   # seconds to re-shoot (limited by the hardware tier)
    resolution: str | None = None     # "640x352", ... (limited by the tier)
    upscale: bool = False             # 2x latent upscale (tier-gated)
    fill: bool = True                 # fill revealed / out-of-frame areas instead of guiding with black


_hw_cache: dict = {"t": 0.0, "rep": None}


def hw_report(refresh: bool = False) -> dict:
    """Hardware report, cached 30 s. Mock mode emulates a tier (RESHOOT_TIER, default 16gb) without probing."""
    if not refresh and _hw_cache["rep"] and _time.time() - _hw_cache["t"] < 30:
        return _hw_cache["rep"]
    if queue.backend.name == "mock":
        t = dict(hardware.tier_by_id(os.environ.get("RESHOOT_TIER", "16gb")))
        rep = {"comfy_url": None, "reachable": False, "hosted": False, "mock": True, "gpu": "mock backend (no GPU used)",
               "vram_total_mb": 0, "vram_free_mb": 0, "tier": t, "caps": hardware.caps(t), "models": [], "missing": [],
               "warnings": ["Mock mode: emulating the " + t["label"] + " tier. Outputs are 2D stand-ins."],
               "can_render": True, "unet": None, "reserve_vram": None, "launch_args": [], "nvidia_smi": None}
    else:
        url = os.environ.get("COMFY_LOCAL_URL", "http://127.0.0.1:8001") if queue.backend.name == "local" \
            else os.environ.get("COMFY_BASE_URL", "https://cloud.comfy.org")
        rep = hardware.detect(url, hosted=queue.backend.name in ("comfy", "proxy"))
    _hw_cache.update(t=_time.time(), rep=rep)
    return rep


def job_opts(req: "JobReq", src_frames: int, src_wh: tuple[int, int] | None = None) -> dict:
    """Validate requested duration/resolution/upscale against the tier; return render options."""
    rep = hw_report()
    t = rep["tier"]
    if not t["supported"]:
        raise HTTPException(400, "This GPU is below the minimum tier: " + t["why"])
    fps = SAFE["fps"]
    aspect = hardware.aspect_of(*src_wh) if src_wh else "landscape"
    auto = hardware.tier_resolutions(t, aspect)
    res = req.resolution or (auto[0] if auto else f"{SAFE['width']}x{SAFE['height']}")
    allowed = hardware.tier_resolutions(t)
    if res not in allowed:
        raise HTTPException(400, f"{res} is not available on this hardware (allowed: {', '.join(allowed)})")
    w, h = (int(x) for x in res.split("x"))
    want = req.duration_s if req.duration_s else SAFE["frames"] / fps
    if want > t["max_seconds"] + 1e-6:
        raise HTTPException(400, f"{want:g} s is longer than this hardware allows ({t['max_seconds']} s)")
    frames = min(int(round(want * fps)), src_frames)
    if req.upscale:
        if t["upscale"] == "no":
            raise HTTPException(400, "2x upscale is not available on this hardware")
        if frames > t["upscale_max_frames"]:
            raise HTTPException(400, f"2x upscale is limited to {t['upscale_max_frames']} frames "
                                     f"({t['upscale_max_frames'] / fps:.1f} s) on this hardware")
    frames = safe_frames(frames, frames) if frames <= t["chunk_frames"] else frames
    opts = {"frames": frames, "width": w, "height": h, "upscale": bool(req.upscale), "fill": bool(req.fill),
            "window": t["chunk_frames"], "overlap": t["overlap"], "tier": t["id"]}
    chunked = req.fill or frames > t["chunk_frames"] or req.upscale or res != f"{SAFE['width']}x{SAFE['height']}" or \
        (rep.get("unet") and rep["unet"] != "ltx-2.5-22b-distilled-Q4_K_M.gguf")
    if chunked:
        opts["chunked"] = True
        if rep.get("unet"):
            opts["unet"] = rep["unet"]
    return opts


@app.get("/api/health")
def health():
    return {"ok": True, "backend": queue.backend.name}


@app.get("/api/limits")
def limits():
    return {**SAFE, "max_input_seconds": MAX_INPUT_SECONDS, "max_input": "3840x2160", "max_angles": 8}


@app.get("/api/hardware")
def get_hardware(refresh: bool = False):
    return hw_report(refresh)


@app.get("/api/presets")
def presets():
    return {"max_angle_deg": MAX_ANGLE_DEG, "presets": [{"id": k, **v} for k, v in PRESETS.items()]}


@app.post("/api/uploads")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"unsupported file type {ext!r}")
    uid = uuid.uuid4().hex[:12]
    dest = queue.uploads / f"{uid}{ext}"
    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                f.close(); dest.unlink(missing_ok=True)
                raise HTTPException(413, "file too large")
            f.write(chunk)
    try:
        info = probe(dest)
    except Exception:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "could not read video")
    dur = info["frames"] / max(info["fps"], 1e-6)
    if dur > MAX_INPUT_SECONDS or info["width"] * info["height"] > MAX_INPUT_PIXELS:
        dest.unlink(missing_ok=True)
        raise HTTPException(413, f"clip too large: max {MAX_INPUT_SECONDS:.0f} s and 3840x2160")
    return {"upload_id": uid, "url": f"/api/uploads/{uid}", "bytes": size, "source": info,
            "aspect": hardware.aspect_of(info["width"], info["height"]),
            "notice": render_notice(info)}


def render_notice(info: dict) -> str:
    try:
        t = hw_report()["tier"]
        max_s = t["max_seconds"]
        res = hardware.tier_resolutions(t, hardware.aspect_of(info["width"], info["height"])) or \
            [f"{SAFE['width']}x{SAFE['height']}"]
    except Exception:  # noqa: BLE001
        max_s, res = SAFE["frames"] / SAFE["fps"], [f"{SAFE['width']}x{SAFE['height']}"]
    src_s = info["frames"] / max(info["fps"], 1e-6)
    src = f"{info['width']}x{info['height']}, {src_s:.1f} s"
    return (f"Your clip ({src}): this machine can re-shoot up to {min(max_s, src_s):.1f} s "
            f"at up to {res[-1]}. Pick the length below; the clip is trimmed and resized before it reaches the GPU.")


@app.get("/api/uploads/{uid}")
def get_upload(uid: str):
    p = queue.upload_path(uid) if uid.isalnum() else None
    if not p:
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/uploads/{uid}/depth")
def get_depth(uid: str):
    p = queue.upload_path(uid) if uid.isalnum() else None
    if not p:
        raise HTTPException(404)
    provider = os.environ.get("RESHOOT_DEPTH_PREVIEW", "comfy" if queue.backend.name == "local" else "heuristic")
    return depth_preview(p, queue.uploads / f"{uid}.depth.json", provider,
                         os.environ.get("COMFY_LOCAL_URL", "http://127.0.0.1:8001"))


@app.get("/api/presets/{preset}/path")
def preset_path(preset: str, angle_deg: float = 15.0):
    if preset not in PRESETS:
        raise HTTPException(404)
    return preset_to_json(preset, clamp_strength(angle_deg))


@app.post("/api/jobs")
def create_job(req: JobReq):
    if not req.upload_id.isalnum() or not queue.upload_path(req.upload_id):
        raise HTTPException(404, "upload not found")
    angles = []
    for a in req.angles:
        if a.preset == "custom":
            if not a.path:
                raise HTTPException(400, "custom angle needs a camera path")
            try:
                angles.append(("custom", 0.0, normalize_path(a.path)))
            except (ValueError, KeyError, TypeError) as e:
                raise HTTPException(400, f"invalid camera path: {e}")
        elif a.preset in PRESETS:
            angles.append((a.preset, clamp_strength(a.angle_deg), None))
        else:
            raise HTTPException(400, f"unknown preset {a.preset!r}")
    if any(j.status in ("queued", "running") for j in queue.jobs.values()):
        raise HTTPException(409, "A re-shoot is already running. One job at a time keeps the GPU safe.")
    _info = probe(queue.upload_path(req.upload_id))
    opts = job_opts(req, _info["frames"], (_info["width"], _info["height"]))
    job = queue.submit(req.upload_id, angles, opts)
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = queue.cancel(job_id)
    if not job:
        raise HTTPException(404)
    return job.to_dict()


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    if any(j.status in ("queued", "running") for j in queue.jobs.values() if j.id != job_id):
        raise HTTPException(409, "Another re-shoot is running.")
    job = queue.resume(job_id)
    if not job:
        raise HTTPException(404)
    return job.to_dict()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = queue.get(job_id)
    if not job:
        raise HTTPException(404)
    return job.to_dict()


class SocialReq(BaseModel):
    angle: int = 0                    # index into job.angles
    layout: str = "stacked"           # stacked | wipe
    labels: list[str] = Field(default_factory=lambda: ["Before", "After"], max_length=2)
    credit: str = Field("", max_length=80)


@app.post("/api/jobs/{job_id}/social")
def social_export(job_id: str, req: SocialReq):
    """1080x1920 before/after mp4 for social posts (ffmpeg on the CPU; no GPU)."""
    from .social import export_social
    job = queue.get(job_id)
    if not job:
        raise HTTPException(404)
    if req.layout not in ("stacked", "wipe"):
        raise HTTPException(400, "layout must be stacked or wipe")
    if not (0 <= req.angle < len(job.angles)) or not job.angles[req.angle].output:
        raise HTTPException(400, "that angle has no finished output")
    after = queue.outputs / job.angles[req.angle].output
    before = queue.upload_path(job.upload_id)
    if not before or not after.is_file():
        raise HTTPException(404, "source or output missing")
    labels = (list(req.labels) + ["Before", "After"])[:2]
    name = f"{Path(after.name).stem}_social_{req.layout}.mp4"
    try:
        export_social(before, after, queue.outputs / name, req.layout, (labels[0][:40], labels[1][:40]), req.credit)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"export failed: {e}")
    return {"name": name, "url": f"/api/outputs/{name}", "download": f"/api/outputs/{name}?download=true"}


@app.get("/api/outputs/{name}")
def output(name: str, download: bool = False):
    p = queue.outputs / name
    if "/" in name or ".." in name or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="video/mp4", filename=name if download else None)


dist = ROOT / "frontend" / "dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
