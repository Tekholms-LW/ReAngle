"""Pluggable render backends: mock (ffmpeg), comfy (hosted Comfy API/Cloud), local (ComfyUI + comfy-api-proxy)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Protocol

from .presets import MAX_ANGLE_DEG

ProgressCb = Callable[..., None]   # progress(fraction, detail: str | None = None)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKFLOW = ROOT / "workflows" / "ltx25_reshoot_api.json"


class RenderBackend(Protocol):
    name: str

    def render(self, src: Path, preset: str, angle_deg: float, out: Path, progress: ProgressCb,
               path: str | None = None) -> Path: ...


def probe(src: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json", str(src)],
        capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    num, den = (s.get("r_frame_rate") or "24/1").split("/")
    fps = float(num) / float(den or 1) or 24.0
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps,
            "frames": int(s.get("nb_read_packets") or 0) or 1}


SAFE = {
    "frames": int(os.environ.get("RESHOOT_MAX_FRAMES", "41")),
    "width": int(os.environ.get("RESHOOT_WIDTH", "640")),
    "height": int(os.environ.get("RESHOOT_HEIGHT", "352")),
    "fps": int(os.environ.get("RESHOOT_FPS", "24")),
}


def safe_frames(n_src: int, cap: int) -> int:
    n = min(max(n_src, 1), cap)
    return max(9, (n - 1) // 8 * 8 + 1)          # LTX needs 8n+1 frames


def prepare_clip(src: Path, out: Path, frames: int, width: int, height: int, fps: int) -> Path:
    """Trim + resize BEFORE anything reaches the GPU: first `frames` frames, centre-cropped to WxH at `fps`."""
    vf = f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", vf, "-frames:v", str(frames),
           "-t", f"{frames / fps:.4f}", "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-crf", "16",
           "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def apply_camera(raw: dict, b: dict, path: str | None) -> None:
    """Choose the camera source: preset node, or the JSON-path node (custom path drawn in the editor)."""
    pnode, jnode = b.get("preset", [None])[0], b.get("path_json", [None])[0]
    target = b.get("camera_input")
    if path:
        if not jnode or not target:
            raise RuntimeError("workflow has no custom camera-path node; rebuild it with build_reshoot_workflow.py")
        raw[jnode]["inputs"][b["path_json"][1]] = path
        raw[target[0]]["inputs"][target[1]] = [jnode, 0]
        raw.pop(pnode, None)
    elif jnode:
        raw.pop(jnode, None)


def mock_filter(preset: str, angle_deg: float, w: int, h: int, n: int, fps: float) -> str:
    """A cheap 2D stand-in for a camera move (crop/zoom/pan over time). Not a real re-angle."""
    k = abs(angle_deg) / MAX_ANGLE_DEG          # 0..1
    sgn = 1 if angle_deg >= 0 else -1
    p = f"(on/{max(n - 1, 1)})"                  # 0..1 progress
    zoom, x, y, rot = "1.2", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2", None
    if preset == "push":
        zoom = f"1+{0.35 * k:.4f}*{p}"
    elif preset == "pull":
        zoom = f"1+{0.35 * k:.4f}*(1-{p})"
    elif preset == "truck":
        x = f"(iw-iw/zoom)*(0.5+{0.5 * sgn * k:.4f}*({p}-0.5)*2)"
    elif preset in ("orbit_left", "orbit_right"):
        d = -1 if preset == "orbit_left" else 1
        x = f"(iw-iw/zoom)*(0.5+{0.5 * d * k:.4f}*{p})"
        rot = f"{d * 0.04 * k:.4f}*t"
    elif preset == "low":
        y = f"(ih-ih/zoom)*(0.5+{0.5 * k:.4f})"
    elif preset == "high":
        y = f"(ih-ih/zoom)*(0.5-{0.5 * k:.4f})"
    elif preset == "crane":
        y = f"(ih-ih/zoom)*(1-{p}*{k:.4f})"
    f = f"zoompan=z='{zoom}':x='{x}':y='{y}':d=1:s={w}x{h}:fps={fps:.3f}"
    if rot:
        f += f",rotate=a='{rot}':fillcolor=black"
    return f


class MockBackend:
    name = "mock"

    def render(self, src, preset, angle_deg, out, progress, path=None, opts=None, work=None, cancelled=None):
        opts, cancelled = opts or {}, cancelled or (lambda: False)
        if path:   # mock can't warp; approximate a custom path by an orbit with the same final yaw
            from .camerapath import approx_yaw_deg
            yaw = approx_yaw_deg(path)
            preset, angle_deg = ("orbit_right" if yaw > 0 else "orbit_left"), min(abs(yaw), MAX_ANGLE_DEG)
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found")
        info = probe(src)
        w, h = info["width"] // 2 * 2, info["height"] // 2 * 2
        n = int(opts.get("frames") or info["frames"])
        n = min(n, info["frames"])
        chunks = None
        if opts.get("chunked"):
            from .chunked import plan_chunks
            chunks = plan_chunks(n, int(opts.get("window", 41)), int(opts.get("overlap", 9))).chunks()
        if opts.get("width"):
            w, h = int(opts["width"]) * (2 if opts.get("upscale") else 1), int(opts["height"]) * (2 if opts.get("upscale") else 1)
        vf = mock_filter(preset, angle_deg, w, h, n, info["fps"])
        if opts.get("width"):
            vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}," + vf
        cmd = ["ffmpeg", "-y", "-v", "error", "-progress", "pipe:1", "-i", str(src), "-frames:v", str(n), "-vf", vf,
               "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-preset", "veryfast", "-c:a", "aac", "-shortest", str(out)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert proc.stdout
        for line in proc.stdout:
            if cancelled():
                proc.kill()
                from .chunked import Cancelled
                raise Cancelled("cancelled")
            if line.startswith("frame="):
                try:
                    f = int(line.split("=")[1])
                    detail = None
                    if chunks:
                        k = next((i for i, (s, e) in enumerate(chunks) if f < e), len(chunks) - 1)
                        detail = f"Chunk {k + 1}/{len(chunks)} (frames {chunks[k][0]}-{chunks[k][1] - 1}) · mock"
                    progress(min(0.99, f / n), detail)
                except ValueError:
                    pass
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.read() if proc.stderr else ''}")
        self.last_stats = {"mock": True, "frames": n, "chunks": len(chunks) if chunks else 1}
        progress(1.0)
        return out


class ComfyBackend:
    """Runs the ReAngle workflow through the Comfy SDK (Comfy API v2).

    Same code for hosted Comfy API / Comfy Cloud and for a local ComfyUI behind
    comfy-api-proxy; only COMFY_BASE_URL differs. Input bindings come from the
    workflow's "_reshoot_bindings" block so the graph can change without code changes.
    """

    def __init__(self, name: str, base_url: str | None, api_key: str | None, workflow: Path = DEFAULT_WORKFLOW):
        self.name, self.base_url, self.api_key, self.workflow_path = name, base_url, api_key, workflow

    def _client(self):
        from comfy_sdk import Comfy  # imported lazily so mock mode needs no SDK
        if self.base_url:
            os.environ["COMFY_BASE_URL"] = self.base_url
        return Comfy(api_key=self.api_key, timeout=60.0)

    def render(self, src, preset, angle_deg, out, progress, path=None, opts=None, work=None, cancelled=None):
        raw = json.loads(self.workflow_path.read_text())
        bindings = raw.pop("_reshoot_bindings")
        raw.pop("_comment", None)
        apply_camera(raw, bindings, path)
        with self._client() as client:
            wf = client.workflows.from_json(raw)
            asset = client.assets.from_file(src)
            wf.set_input(*bindings["video"], asset)
            if not path:
                wf.set_input(*bindings["preset"], preset)
            if not path:
                wf.set_input(*bindings["angle_deg"], float(angle_deg))
            progress(0.05)
            job = client.submit(wf, metadata={"app": "reangle", "preset": preset})
            job = job.result()
            progress(0.95)
            videos = [o for o in job.outputs if (o.content_type or "").startswith("video")] or list(job.outputs)
            if not videos:
                raise RuntimeError("workflow produced no outputs")
            videos[0].to_file(out)
        progress(1.0)
        return out


class NativeComfyBackend:
    """Talks to a plain ComfyUI server (e.g. Comfy Desktop on 127.0.0.1:8001) over its native HTTP API:
    POST /upload/image (any file -> input dir), POST /prompt, GET /history/{id}, GET /view.
    Progress is estimated from elapsed time vs RESHOOT_EXPECTED_SECONDS (ComfyUI only streams
    step progress over websocket)."""

    name = "local"

    def __init__(self, base_url: str, workflow: Path = DEFAULT_WORKFLOW, max_frames: int = 41,
                 expected_s: float = 480.0, timeout_s: float = 3600.0):
        self.base = base_url.rstrip("/")
        self.workflow_path, self.max_frames = workflow, max_frames
        self.expected_s, self.timeout_s = expected_s, timeout_s

    def _http(self, method: str, path: str, data: bytes | None = None, headers: dict | None = None):
        import urllib.request
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers or {})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    def _upload(self, src: Path) -> str:
        import uuid as _uuid
        boundary = _uuid.uuid4().hex
        name = f"reangle_{_uuid.uuid4().hex[:8]}{src.suffix.lower()}"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + src.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        r = json.loads(self._http("POST", "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}))
        return (r.get("subfolder") + "/" if r.get("subfolder") else "") + r["name"]

    def _history(self, pid: str) -> dict:
        return json.loads(self._http("GET", f"/history/{pid}"))

    # hooks the chunked renderer calls; the cloud backend overrides them
    def depth_graph(self, video: str, o) -> tuple[dict, str]:
        """(graph, output node) producing a grey depth image per frame, near = bright."""
        return ({"1": {"class_type": "VHS_LoadVideo", "inputs": {"video": video, "force_rate": o.fps,
                       "custom_width": o.width, "custom_height": o.height, "frame_load_cap": o.frames,
                       "skip_first_frames": 0, "select_every_nth": 1}},
                 "10": {"class_type": "LoadVideoDepthAnythingModel", "inputs": {"model": "video_depth_anything_vits.pth"}},
                 "11": {"class_type": "VideoDepthAnythingProcess", "inputs": {"vda_model": ["10", 0], "images": ["1", 0],
                        "input_size": 518, "max_res": 960, "precision": "fp16"}},
                 "12": {"class_type": "VideoDepthAnythingOutput", "inputs": {"depths": ["11", 0], "colormap": "gray"}},
                 "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": "reangle/depthseq"}}},
                "13")

    def patch_graph(self, g: dict) -> dict:
        return g

    def frames_for(self, src: Path) -> int:
        return safe_frames(probe(src)["frames"], self.max_frames)

    def preflight(self) -> dict:
        """Refuse to submit unless ComfyUI is idle and enough VRAM is free (protects the display driver)."""
        q = json.loads(self._http("GET", "/queue"))
        if q.get("queue_running") or q.get("queue_pending"):
            raise RuntimeError("ComfyUI is busy with another job; try again when it finishes.")
        stats = json.loads(self._http("GET", "/system_stats"))
        argv = (stats.get("system") or {}).get("argv") or []
        if os.environ.get("RESHOOT_REQUIRE_RESERVE_VRAM", "1") == "1":
            res = 0.0
            for i, a in enumerate(argv):
                if a == "--reserve-vram" and i + 1 < len(argv):
                    try:
                        res = float(argv[i + 1])
                    except ValueError:
                        pass
            if res < 2 and "--lowvram" not in argv and "--novram" not in argv:
                raise RuntimeError("ComfyUI is not started with --reserve-vram 2 (or more). Without it the "
                                   "22B model fills the whole GPU and can freeze the Windows display driver. "
                                   "Add '--reserve-vram 3' to Comfy Desktop launch args and restart it.")
        dev = (stats.get("devices") or [{}])[0]
        free_mb = int(dev.get("vram_free", 0)) // (1 << 20)
        need = int(os.environ.get("RESHOOT_MIN_FREE_VRAM_MB", "12000"))
        deadline = time.time() + float(os.environ.get("RESHOOT_PREFLIGHT_WAIT_S", "45"))
        while free_mb < need and time.time() < deadline:   # unloading after /free can take a few seconds
            time.sleep(3)
            dev = (json.loads(self._http("GET", "/system_stats")).get("devices") or [{}])[0]
            free_mb = int(dev.get("vram_free", 0)) // (1 << 20)
        if free_mb < need:
            raise RuntimeError(f"Only {free_mb} MB of GPU memory is free (need {need} MB). "
                               "Close GPU-heavy apps (games, video, other AI tools) and retry.")
        return {"vram_free_mb": free_mb, "vram_total_mb": int(dev.get("vram_total", 0)) // (1 << 20)}

    # ---- shared plumbing (also used by chunked.ChunkedRenderer)
    last_peak_mb: int = 0
    cancel_check = staticmethod(lambda: False)

    def free(self) -> None:
        if os.environ.get("RESHOOT_FREE_BEFORE", "1") == "1":
            self._http("POST", "/free", json.dumps({"unload_models": True, "free_memory": True}).encode(),
                       {"Content-Type": "application/json"})
            time.sleep(2)

    def vram_used_mb(self) -> tuple[int, int] | None:
        """(used, total) MB. nvidia-smi when ComfyUI is local (sees the whole card), else /system_stats."""
        from .hardware import nvidia_smi
        if any(h in self.base for h in ("127.0.0.1", "localhost")):
            s = nvidia_smi()
            if s:
                return s["used_mb"], s["total_mb"]
        try:
            d = (json.loads(self._http("GET", "/system_stats")).get("devices") or [{}])[0]
            tot = int(d.get("vram_total", 0)) >> 20
            return tot - (int(d.get("vram_free", 0)) >> 20), tot
        except Exception:  # noqa: BLE001
            return None

    def run_prompt(self, raw: dict, out_node: str, kind: str = "video", label: str = "", span=(0.02, 0.95),
                   expected_s: float | None = None, progress: ProgressCb | None = None) -> list[bytes]:
        """Submit, poll, watch VRAM (abort before the card fills up), honour cancel, fetch outputs."""
        import urllib.parse
        progress = progress or getattr(self, "_progress", None) or (lambda p, d=None: None)
        expected_s = expected_s or self.expected_s
        r = json.loads(self._http("POST", "/prompt", json.dumps({"prompt": raw}).encode(),
                                  {"Content-Type": "application/json"}))
        if r.get("node_errors"):
            raise RuntimeError(f"ComfyUI rejected workflow: {r['node_errors']}")
        pid, t0 = r["prompt_id"], time.time()
        self.last_peak_mb = 0
        from .vram import VramSampler
        samp = VramSampler() if any(h in self.base for h in ("127.0.0.1", "localhost")) else None
        fast = bool(samp and samp.start())
        try:
            return self._poll(pid, t0, samp if fast else None, out_node, kind, label, span, expected_s, progress)
        finally:
            if samp:
                samp.stop()

    def _poll(self, pid, t0, samp, out_node, kind, label, span, expected_s, progress):
        import urllib.parse
        from .vram import VramSampler
        slow = VramSampler(strikes=2)   # cross-check every ~2 s via vram_used_mb (the only source when remote)
        while True:
            for _ in range(10):                          # check the fast sampler every 0.2 s, ComfyUI every 2 s
                time.sleep(0.2 if samp else 0.1)
                if samp and samp.trip:
                    break
            v = self.vram_used_mb()
            if v:
                slow.feed(v[0], v[1])
            self.last_peak_mb = max(self.last_peak_mb, slow.peak, samp.peak if samp else 0)
            w = samp if samp and samp.trip else slow
            if w.trip:
                self._http("POST", "/interrupt", b"{}", {"Content-Type": "application/json"})
                self.free()
                raise RuntimeError(f"Stopped: {w.trip}. Use a shorter chunk / lower resolution.")
            if self.cancel_check():
                self._http("POST", "/interrupt", b"{}", {"Content-Type": "application/json"})
                self.free()
                from .chunked import Cancelled
                raise Cancelled("cancelled")
            h = self._history(pid)
            if pid in h and h[pid].get("status", {}).get("completed") is not None and \
                    h[pid]["status"].get("status_str") in ("success", "error"):
                break
            if time.time() - t0 > self.timeout_s:
                raise TimeoutError("ComfyUI job timed out")
            el = time.time() - t0
            progress(min(span[1], span[0] + (span[1] - span[0]) * min(0.97, el / expected_s)),
                     f"{label} · {int(el)} s · GPU {self.last_peak_mb} MB peak" if label else None)
        st = h[pid]["status"]
        if st.get("status_str") != "success":
            msg = [m[1].get("exception_message", "") for m in st.get("messages", []) if m[0] == "execution_error"]
            raise RuntimeError(f"ComfyUI execution failed: {msg[:1]}")
        outs = h[pid]["outputs"].get(out_node, {})
        files = ([f for f in (outs.get("gifs") or []) if f["filename"].endswith(".mp4")] if kind == "video"
                 else list(outs.get("images") or []))
        if not files:
            raise RuntimeError(f"no {kind} output from ComfyUI")
        if kind == "video":
            files = files[-1:]
        res = []
        for f in files:
            q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                        "type": f.get("type", "output")})
            res.append(self._http("GET", f"/view?{q}"))
        return res

    def render(self, src, preset, angle_deg, out, progress, path=None, opts: dict | None = None,
               work: Path | None = None, cancelled=None):
        self._progress = progress
        self.cancel_check = cancelled or (lambda: False)
        opts = opts or {}
        if opts.get("chunked") or opts.get("upscale") or opts.get("unet"):
            return self._render_chunked(src, preset, angle_deg, out, progress, path, opts, work)
        raw = json.loads(self.workflow_path.read_text())
        b = raw.pop("_reshoot_bindings")
        raw.pop("_comment", None)
        apply_camera(raw, b, path)
        out_node = b.pop("output_node", "90")
        frames = self.frames_for(src)
        self.free()
        self.preflight()
        small = Path(out).with_name(Path(out).stem + "_input.mp4")
        prepare_clip(src, small, frames, SAFE["width"], SAFE["height"], SAFE["fps"])
        vals = {"video": self._upload(small), "preset": preset, "angle_deg": float(angle_deg),
                "frames": frames, "num_frames": frames, "num_frames_json": frames,
                "width": SAFE["width"], "height": SAFE["height"]}
        if path:
            vals.pop("preset"); vals.pop("angle_deg"); vals.pop("num_frames")
        for k in [k for k, v in b.items() if isinstance(v, list) and v and v[0] not in raw]:
            b.pop(k)                                     # binding to a node removed by apply_camera
        for k, v in vals.items():
            if k in b:
                node, field = b[k]
                raw[node]["inputs"][field] = v
        progress(0.02)
        data = self.run_prompt(raw, out_node, progress=progress)
        Path(out).write_bytes(data[0])
        self.last_stats = {"peak_vram_mb": self.last_peak_mb}
        progress(1.0)
        return out

    def _render_chunked(self, src, preset, angle_deg, out, progress, path, opts, work):
        from .chunked import ChunkedRenderer, RenderOpts
        w, h = opts.get("width", SAFE["width"]), opts.get("height", SAFE["height"])
        fps = SAFE["fps"]
        frames = int(opts.get("frames") or SAFE["frames"])
        work = work or Path(out).with_suffix("")
        clip = work / "input.mp4"
        work.mkdir(parents=True, exist_ok=True)
        if not clip.exists():
            prepare_clip(src, clip, frames, w, h, fps)
        ro = RenderOpts(width=w, height=h, fps=fps, frames=frames, window=int(opts.get("window", 41)),
                        overlap=int(opts.get("overlap", 9)), upscale=bool(opts.get("upscale")), unet=opts.get("unet"),
                        fill=bool(opts.get("fill", True)))
        r = ChunkedRenderer(self, work, ro, progress, self.cancel_check)
        r.run(clip, preset, angle_deg, path, Path(out))
        self.last_stats = r.stats
        progress(1.0)
        return out


class CloudComfyBackend(NativeComfyBackend):
    """Comfy Cloud (Comfy Developer Platform) through its ComfyUI-compatible /api endpoints, X-API-Key auth.

    Comfy Cloud runs LTX-2.5 at full precision but has neither our custom nodes nor the LTX-2.5
    Layout-to-Render IC-LoRA, so: depth runs on Cloud with Depth Anything V2 Small (per frame), the camera
    warp + background-aware fill run in the app (CPU, same code as the ReAngle nodes), and Cloud runs the LTX
    generation graph with the warped guide + regen mask uploaded as inputs. No VRAM watchdog is needed
    (Cloud GPUs); the job is interrupted on cancel and credits are reported by /api/billing/usage/timeseries.
    """

    name = "cloud"
    UNET = os.environ.get("RESHOOT_CLOUD_UNET", "ltx-2.5-22b-distilled-transformer-bf16.safetensors")
    LORA = os.environ.get("RESHOOT_CLOUD_LORA", "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors")
    DEPTH = os.environ.get("RESHOOT_CLOUD_DEPTH", "depth_anything_v2_vits.pth")   # Apache-2.0 (Small)

    def __init__(self, api_key: str, base_url: str | None = None, **kw):
        super().__init__((base_url or "https://cloud.comfy.org").rstrip("/") + "/api", **kw)
        self._key = api_key
        self.expected_s = float(os.environ.get("RESHOOT_EXPECTED_SECONDS", "240"))

    def __repr__(self):                       # never leak the key in logs
        return f"CloudComfyBackend({self.base!r})"

    def _http(self, method, path, data=None, headers=None):
        h = dict(headers or {}); h["X-API-Key"] = self._key
        return super()._http(method, path, data, h)

    def _history(self, pid):
        j = json.loads(self._http("GET", f"/jobs/{pid}"))
        st = j.get("status")
        if st not in ("completed", "failed", "cancelled"):
            return {}
        err = j.get("execution_error") or {}
        msgs = [["execution_error", {"exception_message": err.get("exception_message", st)}]] if st != "completed" else []
        return {pid: {"status": {"completed": st == "completed", "status_str": "success" if st == "completed" else "error",
                                 "messages": msgs}, "outputs": j.get("outputs") or {}}}

    def free(self):
        return None

    def vram_used_mb(self):
        return None

    def preflight(self):
        return {"cloud": True}

    def balance_cents(self) -> int | None:
        try:
            d = json.loads(self._http("GET", "/billing/usage/timeseries"))
            return int(((d.get("summary") or d).get("balance") or {}).get("amount_cents"))
        except Exception:  # noqa: BLE001
            return None

    def render(self, src, preset, angle_deg, out, progress, path=None, opts=None, work=None, cancelled=None):
        opts = dict(opts or {}); opts["chunked"] = True      # Cloud lacks the in-graph ReAngle nodes
        return super().render(src, preset, angle_deg, out, progress, path, opts, work, cancelled)

    def depth_graph(self, video, o):
        return ({"1": {"class_type": "VHS_LoadVideo", "inputs": {"video": video, "force_rate": o.fps,
                       "custom_width": o.width, "custom_height": o.height, "frame_load_cap": o.frames,
                       "skip_first_frames": 0, "select_every_nth": 1}},
                 "11": {"class_type": "DepthAnythingV2Preprocessor", "inputs": {"image": ["1", 0],
                        "ckpt_name": self.DEPTH, "resolution": 512}},
                 "13": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "reangle/depthseq"}}},
                "13")

    def patch_graph(self, g):
        for n in g.values():
            if n["class_type"] in ("UnetLoaderGGUF", "UNETLoader"):
                n["class_type"] = "UNETLoader"
                n["inputs"] = {"unet_name": self.UNET, "weight_dtype": "default"}
            elif n["class_type"] == "LTXICLoRALoaderModelOnly":
                n["inputs"]["lora_name"] = self.LORA
        return g


def make_backend(kind: str | None = None) -> RenderBackend:
    kind = (kind or os.environ.get("RESHOOT_BACKEND", "mock")).lower()
    if kind == "mock":
        return MockBackend()
    if kind == "comfy-v2":   # experimental: the asset-based v2 workflow API client (needs ReAngle nodes on the server)
        key = os.environ.get("COMFY_API_KEY")
        if not key:
            raise RuntimeError("RESHOOT_BACKEND=comfy-v2 needs COMFY_API_KEY")
        return ComfyBackend("comfy", os.environ.get("COMFY_BASE_URL"), key)
    if kind == "local":   # plain ComfyUI (native /prompt API), e.g. Comfy Desktop on :8001
        return NativeComfyBackend(os.environ.get("COMFY_LOCAL_URL", "http://127.0.0.1:8001"),
                                  Path(os.environ.get("RESHOOT_WORKFLOW", DEFAULT_WORKFLOW)),
                                  int(os.environ.get("RESHOOT_MAX_FRAMES", "41")),
                                  float(os.environ.get("RESHOOT_EXPECTED_SECONDS", "480")))
    if kind in ("comfy", "cloud"):   # Comfy Cloud via its ComfyUI-compatible API (no custom nodes needed on Cloud)
        key = os.environ.get("COMFY_API_KEY")
        if not key:
            raise RuntimeError("RESHOOT_BACKEND=comfy needs COMFY_API_KEY (Comfy Cloud paid plan)")
        return CloudComfyBackend(key, os.environ.get("COMFY_BASE_URL"))
    if kind == "proxy":   # self-hosted ComfyUI behind comfy-api-proxy (Comfy API v2 via comfy-sdk)
        return ComfyBackend("proxy", os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8189"),
                            os.environ.get("COMFY_API_KEY") or "local")
    raise ValueError(f"unknown backend {kind!r}")
