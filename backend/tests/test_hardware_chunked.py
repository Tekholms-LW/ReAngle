import json, os, subprocess, tempfile, time
from pathlib import Path

import numpy as np
import pytest

from reshoot import hardware
from reshoot.chunked import ChunkedRenderer, RenderOpts, blend_into, plan_chunks, read_frames


# ---------- tiers
@pytest.mark.parametrize("mb,tid", [(8192, "under12"), (11000, "under12"), (12288, "16gb"), (16303, "16gb"),
                                    (24564, "24gb"), (32607, "32gb"), (49140, "48gb"), (81920, "48gb")])
def test_tier_for_vram(mb, tid):
    assert hardware.tier_for_vram(mb)["id"] == tid


def test_tiers_monotonic():
    sup = [t for t in hardware.TIERS if t["supported"]]
    for a, b in zip(sup, sup[1:]):
        assert b["max_seconds"] >= a["max_seconds"] and b["chunk_frames"] >= a["chunk_frames"]
        assert len(b["resolutions"]) >= len(a["resolutions"])
    for t in sup:
        assert (t["chunk_frames"] - 1) % 8 == 0 and (t["overlap"] - 1) % 8 == 0
        assert all(u in hardware.UNETS for u in t["unets"])


def test_detect_unreachable_and_models(monkeypatch):
    rep = hardware.detect("http://127.0.0.1:9", hosted=False, use_smi=False, files={})
    assert not rep["reachable"] and not rep["can_render"]
    stats = {"system": {"argv": ["main.py", "--reserve-vram", "3"], "pytorch_version": "2.8.0+cu129",
                        "ram_total": 64 << 30, "ram_free": 32 << 30},
             "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 5080 : cudaMallocAsync",
                          "vram_total": 17066033152, "vram_free": 15 << 30}]}
    monkeypatch.setattr(hardware, "_get", lambda base, path, timeout=8.0: stats)
    files = {}
    for m in hardware.SHARED_MODELS:
        files.setdefault(m["loader"], []).append(m["file"])
    files[hardware.UNET_GGUF] = ["ltx-2.5-22b-distilled-Q4_K_M.gguf"]
    rep = hardware.detect("http://127.0.0.1:8001", hosted=False, use_smi=False, files=files)
    assert rep["tier"]["id"] == "16gb" and rep["unet"].endswith("Q4_K_M.gguf") and rep["can_render"]
    assert rep["gpu"] == "NVIDIA GeForce RTX 5080" and rep["cuda"] == "129" and rep["reserve_vram"] == 3
    stats["system"]["argv"] = ["main.py"]
    files.pop(("CLIPLoader", "clip_name"))
    rep = hardware.detect("http://127.0.0.1:8001", hosted=False, use_smi=False, files=files)
    assert not rep["can_render"] and any("reserve-vram" in w for w in rep["warnings"])
    assert "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors" in rep["missing"]
    assert all(m["url"].startswith("https://huggingface.co/") for m in rep["models"])
    assert "NOT ready to render" in hardware.summary_text(rep)


def test_hosted_defaults_high_tier(monkeypatch):
    monkeypatch.setattr(hardware, "_get", lambda base, path, timeout=8.0: {"system": {}, "devices": []})
    rep = hardware.detect("https://cloud.example", hosted=True, use_smi=False, files={})
    assert rep["tier"]["id"] == "48gb" and "Hosted" in rep["tier"]["why"]


# ---------- chunk planning + blending
@pytest.mark.parametrize("total,win,ov", [(120, 41, 9), (41, 41, 9), (30, 41, 9), (240, 57, 9), (480, 121, 17), (100, 97, 17)])
def test_plan_covers_with_overlap(total, win, ov):
    p = plan_chunks(total, win, ov)
    ch = p.chunks()
    assert ch[0][0] == 0 and ch[-1][1] == p.total
    assert all(e - s == p.window and (e - s - 1) % 8 == 0 for s, e in ch)
    for (s0, e0), (s1, e1) in zip(ch, ch[1:]):
        assert e0 - s1 >= ov            # enough overlap to condition on the previous output
    if total <= win:
        assert len(ch) == 1 and p.total <= total


def test_blend_is_continuous():
    acc = np.zeros((20, 2, 2, 3), np.float32)
    f = blend_into(acc, 0, np.full((12, 2, 2, 3), 10, np.float32), 0)
    f = blend_into(acc, f, np.full((12, 2, 2, 3), 20, np.float32), 8)
    assert f == 20 and acc[:8].max() == 10 and acc[12:].min() == 20
    ramp = acc[8:12, 0, 0, 0]
    assert np.all(np.diff(ramp) > 0) and 10 < ramp[0] < ramp[-1] < 20


# ---------- full chunked pipeline against a fake ComfyUI (no GPU)
def _clip(p: Path, secs: float, fps=24, size="160x96"):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}:duration={secs}",
                    "-f", "lavfi", "-i", f"sine=frequency=440:duration={secs}", "-shortest",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(p)], check=True)
    return p


class FakeComfy:
    def __init__(self, tmp: Path, upscale=False):
        self.tmp, self.files, self.prompts, self.frees, self.last_peak_mb, self.upscale = tmp, {}, [], 0, 1234, upscale

    def _upload(self, p):
        name = f"u{len(self.files)}{Path(p).suffix}"
        self.files[name] = Path(p)
        return name

    def free(self):
        self.frees += 1

    def preflight(self):
        return {}

    def run_prompt(self, g, out_node, kind="video", label="", span=None, expected_s=None):
        self.prompts.append(g)
        src = self.files[g["1"]["inputs"]["video"]]
        if kind == "images":    # depth: gray ramp PNGs, one per frame
            n = g["1"]["inputs"]["frame_load_cap"]
            d = self.tmp / f"d{len(self.prompts)}"; d.mkdir()
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "gradients=size=160x96:duration=10:rate=24",
                            "-frames:v", str(n), "-pix_fmt", "gray", str(d / "%03d.png")], check=True)
            return [f.read_bytes() for f in sorted(d.glob("*.png"))]
        assert "23" in g and g["90"]["inputs"].get("audio") is None
        if self.upscale:
            assert "71" in g
            o = self.tmp / f"up{len(self.prompts)}.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", "scale=iw*2:ih*2", str(o)], check=True)
            return [o.read_bytes()]
        return [src.read_bytes()]


def test_chunked_pipeline_fake_comfy(tmp_path):
    clip = _clip(tmp_path / "in.mp4", 5)
    fc = FakeComfy(tmp_path)
    msgs = []
    r = ChunkedRenderer(fc, tmp_path / "work", RenderOpts(width=160, height=96, frames=120, window=41, overlap=9, color_match=False),
                        lambda p, d=None: msgs.append((p, d)), lambda: False)
    out = r.run(clip, "orbit_left", 20, None, tmp_path / "out.mp4")
    v = read_frames(out, 160, 96)
    assert len(v) == 120
    chunk_prompts = [g for g in fc.prompts if "23" in g]
    assert len(chunk_prompts) == 4
    assert "24" not in chunk_prompts[0] and chunk_prompts[0]["51"]["inputs"]["bypass"] is False
    for g in chunk_prompts[1:]:     # later chunks continue from the previous output
        assert g["24"]["inputs"]["frame_load_cap"] == 9 and g["51"]["inputs"]["bypass"] is False and g["2"]["inputs"]["image"] == ["23", 0]  # re-anchored to the warped source
        assert g["55"]["class_type"] == "LTXVAddGuide"
    assert any(d and d.startswith("Chunk 4/4") for _, d in msgs)
    a = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name",
                        "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
    assert "aac" in a                     # original audio muxed back
    # source passthrough => blended output ~= source (crossfades are seamless)
    s = read_frames(tmp_path / "work" / "input.mp4" if (tmp_path / "work" / "input.mp4").exists() else clip, 160, 96)[:120]
    assert np.abs(v.astype(int) - s.astype(int)).mean() < 6
    guide = read_frames(tmp_path / "work" / "guide_fill.mp4", 160, 96)      # fill is on by default
    assert len(guide) == 120 and np.abs(guide[-1].astype(int) - s[-1].astype(int)).mean() > 3   # camera moved
    assert (guide.max(-1) < 8).mean() <= (s.max(-1) < 8).mean() + 0.01         # no black holes left in the guide
    assert (tmp_path / "work" / "regen_mask.mp4").exists()

    # resume: finished chunks are reused, nothing is re-rendered
    n_before = len(fc.prompts)
    (tmp_path / "work" / "chunk_03.mp4").unlink()
    r2 = ChunkedRenderer(fc, tmp_path / "work", RenderOpts(width=160, height=96, frames=120, window=41, overlap=9, color_match=False),
                         lambda p, d=None: None, lambda: False)
    r2.run(clip, "orbit_left", 20, None, tmp_path / "out2.mp4")
    assert len(fc.prompts) - n_before == 1    # only the missing chunk (depth + guide cached)


def test_chunked_upscale_and_cancel(tmp_path):
    clip = _clip(tmp_path / "in.mp4", 1.1)
    fc = FakeComfy(tmp_path, upscale=True)
    r = ChunkedRenderer(fc, tmp_path / "w", RenderOpts(width=160, height=96, frames=25, window=41, overlap=9, upscale=True),
                        lambda p, d=None: None, lambda: False)
    out = r.run(clip, "push", 15, None, tmp_path / "up.mp4")
    assert len(read_frames(out, 320, 192)) == 25
    from reshoot.chunked import Cancelled
    r = ChunkedRenderer(fc, tmp_path / "w2", RenderOpts(width=160, height=96, frames=25), lambda p, d=None: None, lambda: True)
    with pytest.raises(Cancelled):
        r.run(clip, "push", 15, None, tmp_path / "c.mp4")


def test_custom_path_global_poses():
    from reshoot.chunked import global_poses
    spec = json.dumps({"version": 1, "pivot_depth": 3, "ease": "linear", "freeze_time": True,
                       "keyframes": [{"t": 0, "pos": [0, 0, 0], "look": [0, 0, 3]}, {"t": 1, "pos": [-0.5, 0, 0], "look": [0, 0, 3]}]})
    P, freeze = global_poses("custom", 0, spec, 120)
    assert P.shape == (120, 4, 4) and freeze and np.allclose(P[0], np.eye(4))


def test_installed_files_both_formats(monkeypatch):
    def fake(base, path, timeout=8.0):
        node = path.rsplit("/", 1)[1]
        if node == "CLIPLoader":
            return {node: {"input": {"required": {"clip_name": [["a.safetensors"]]}}}}
        return {node: {"input": {"required": {k: ["COMBO", {"options": ["b.safetensors"]}]
                                              for k in ("unet_name", "vae_name", "lora_name", "model_name", "model")}}}}
    monkeypatch.setattr(hardware, "_get", fake)
    f = hardware.installed_files("http://x")
    assert f[("CLIPLoader", "clip_name")] == ["a.safetensors"] and f[("LTXICLoRALoaderModelOnly", "lora_name")] == ["b.safetensors"]


def test_prefill_fills_from_background_not_foreground():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "comfyui-reshoot-nodes"))
    from reshoot_nodes.reproject import prefill_holes
    img = np.full((64, 64, 3), 0.7, np.float32); dep = np.full((64, 64), 8.0)
    img[10:50, 10:30] = 0.1; dep[10:50, 10:30] = 2.0           # near dark subject
    hole = np.zeros((64, 64), np.float32); hole[10:50, 30:38] = 1; hole[:, :3] = 1; hole[:4] = 1
    img[hole > 0] = 0
    out, m = prefill_holes(img, hole, dep)
    assert abs(out[30, 34, 0] - 0.7) < 0.05                       # disocclusion -> background, no dark ghost
    assert abs(out[30, 1, 0] - 0.7) < 0.05 and abs(out[1, 50, 0] - 0.7) < 0.05   # border wedges edge-extended
    assert m[30, 34] > 0.99 and m[60, 60] == 0


def test_vram_sampler_catches_spikes_and_sustained():
    from reshoot.vram import VramSampler
    v = VramSampler(); v.feed(16050, 16303, 0); assert v.trip and "spiked" in v.trip
    v = VramSampler(); v.feed(15846, 16303, 0); assert not v.trip and v.peak == 15846
    v = VramSampler(); [v.feed(15900, 16303, t * 0.2) for t in range(6)]; assert v.trip
    v = VramSampler(); [v.feed(15900 if t % 2 else 9000, 16303, t * 0.2) for t in range(10)]; assert not v.trip


def _soc_clip(path, size, seconds=1.5, audio=True):
    import subprocess
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=24:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)], check=True)
    return path


def test_social_export_any_aspect(tmp_path):
    import json
    import subprocess
    import pytest
    from reshoot.social import CREDIT_PRESET, export_social
    for size in ("448x448", "640x352", "352x640"):
        before = _soc_clip(tmp_path / f"b_{size}.mp4", size, 2.0, audio=True)
        after = _soc_clip(tmp_path / f"a_{size}.mp4", size, 1.5, audio=False)
        for layout in ("stacked", "wipe"):
            out = export_social(before, after, tmp_path / f"o_{size}_{layout}.mp4", layout, ("Original", "Re-shot"),
                                CREDIT_PRESET)
            j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                           "stream=codec_type,codec_name,width,height,pix_fmt:format=duration",
                                           "-of", "json", str(out)], capture_output=True, text=True).stdout)
            v = next(s for s in j["streams"] if s["codec_type"] == "video")
            a = [s for s in j["streams"] if s["codec_type"] == "audio"]
            assert (v["width"], v["height"], v["codec_name"], v["pix_fmt"]) == (1080, 1920, "h264", "yuv420p")
            assert a and a[0]["codec_name"] == "aac"
            assert abs(float(j["format"]["duration"]) - 1.5) < 0.15     # trimmed to the re-shoot length
            head = out.read_bytes()[:4096]
            assert head.find(b"moov") != -1 and head.find(b"moov") < max(head.find(b"mdat"), 1 << 30)  # faststart
    with pytest.raises(ValueError):
        export_social(before, after, tmp_path / "x.mp4", "diagonal")


def test_border_mirror_never_copies_the_subject():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "comfyui-reshoot-nodes"))
    from reshoot_nodes.reproject import prefill_holes
    img = np.full((64, 64, 3), 0.7, np.float32); dep = np.full((64, 64), 8.0)
    img[:, 12:30] = 0.05; dep[:, 12:30] = 2.0         # dark near subject right next to the left border gap
    hole = np.zeros((64, 64), np.float32); hole[:, :10] = 1; img[hole > 0] = 0
    out, _ = prefill_holes(img, hole, dep)
    assert out[32, :6, 0].min() > 0.5                   # gap filled with background, not a mirrored dark subject
