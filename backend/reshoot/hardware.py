"""Hardware detection + capability tiers.

Reads ComfyUI /system_stats (+ /object_info for installed model files) and, when the backend runs on the
same machine as the GPU, `nvidia-smi`. Works against a remote/hosted ComfyUI too: hosted endpoints that
hide their hardware are treated as RESHOOT_HOSTED_TIER (default "48gb").

Tier assumptions (documented in README "Hardware tiers"):
  * LTX-2.5 22B weights dominate VRAM: Q4_K_M 13.6 GB, Q6_K 17.8 GB, int8 21.5 GB, bf16 42 GB (HF file sizes).
    ComfyUI streams/offloads weights that don't fit, so the *working set* (activations: ~ width*height*frames
    tokens) is what pushes a card over the edge. We cap tokens per chunk per tier from what is proven safe
    on a 16 GB RTX 5080 (640x352x41 with --reserve-vram 3) and scale roughly linearly with free VRAM.
  * --reserve-vram keeps headroom for the Windows desktop/driver (the 5080 hit a display-driver reset
    without it). Bigger cards need less relative headroom.
  * The 2x latent upscale quadruples tokens in stage 2. On 16 GB it is only allowed for short clips
    (<= 25 frames) and runs slower; 24 GB+ allow it per chunk.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request

GB = 1024
HF = {
    "unsloth": "https://huggingface.co/unsloth/LTX-2.5-GGUF",
    "ltx": "https://huggingface.co/Lightricks/LTX-2.5",
    "layout": "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render",
    "union": "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
    "vda": "https://huggingface.co/depth-anything/Video-Depth-Anything-Small",
}
# (loader node, input name) -> where ComfyUI lists installed files in /object_info
UNET_GGUF = ("UnetLoaderGGUF", "unet_name")
UNET_ST = ("UNETLoader", "unet_name")

SHARED_MODELS = [
    {"role": "text encoder", "file": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "gb": 15.4,
     "folder": "text_encoders", "loader": ("CLIPLoader", "clip_name"), "url": HF["ltx"], "required": True},
    {"role": "video VAE", "file": "ltx-2.5-video-vae-bf16.safetensors", "gb": 1.4, "folder": "vae",
     "loader": ("VAELoader", "vae_name"), "url": HF["ltx"], "required": True},
    {"role": "audio VAE", "file": "ltx-2.5-audio-vae-bf16.safetensors", "gb": 0.3, "folder": "vae",
     "loader": ("VAELoader", "vae_name"), "url": HF["ltx"], "required": True},
    {"role": "IC-LoRA layout-to-render", "file": "ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors", "gb": 1.3,
     "folder": "loras", "loader": ("LTXICLoRALoaderModelOnly", "lora_name"), "url": HF["layout"], "required": True},
    {"role": "2x latent upscaler", "file": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", "gb": 1.0,
     "folder": "latent_upscale_models", "loader": ("LatentUpscaleModelLoader", "model_name"), "url": HF["ltx"],
     "required": False},
    {"role": "depth (Video Depth Anything S, Apache-2.0)", "file": "video_depth_anything_vits.pth", "gb": 0.1,
     "folder": "videodepthanything", "loader": ("LoadVideoDepthAnythingModel", "model"), "url": HF["vda"],
     "required": True},
]
UNETS = {   # file -> (loader, size GB, url)
    "ltx-2.5-22b-distilled-Q4_K_M.gguf": (UNET_GGUF, 13.6, HF["unsloth"]),
    "ltx-2.5-22b-distilled-Q5_K_M.gguf": (UNET_GGUF, 15.6, HF["unsloth"]),
    "ltx-2.5-22b-distilled-Q6_K.gguf": (UNET_GGUF, 17.8, HF["unsloth"]),
    "ltx-2.5-22b-distilled-Q8_0.gguf": (UNET_GGUF, 22.8, HF["unsloth"]),
    "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors": (UNET_ST, 21.5, HF["ltx"]),
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors": (UNET_ST, 42.0, HF["ltx"]),
}

# min VRAM GB, name, preferred unets (first installed wins), resolutions, frames/chunk, overlap, max seconds,
# upscale policy, required --reserve-vram
TIERS = [
    {"id": "under12", "min_gb": 0, "label": "Under 12 GB", "supported": False, "unets": [],
     "resolutions": [], "chunk_frames": 0, "overlap": 0, "max_seconds": 0, "upscale": "no", "upscale_max_frames": 0,
     "reserve_vram": 3,
     "why": "The 22B LTX model plus activations does not fit safely; this risks freezing the display driver. "
            "Use a hosted ComfyUI (RESHOOT_BACKEND=comfy) or a bigger GPU."},
    {"id": "16gb", "min_gb": 12, "label": "12-16 GB (e.g. RTX 5080 / 4080)", "supported": True,
     "unets": ["ltx-2.5-22b-distilled-Q4_K_M.gguf"],
     "resolutions": ["640x352"], "chunk_frames": 33, "overlap": 9, "max_seconds": 5, "upscale": "short",
     "upscale_max_frames": 25, "reserve_vram": 3,
     "why": "Q4 weights stream in; 640x352 x 33 frames per chunk keeps ~0.8 GB headroom on a 16 GB card (41-frame "
            "chunks measured 15.6 GB of 16.3 GB incl. the Windows desktop). "
            "Longer clips are rendered in overlapping chunks. 2x upscale only for short clips (slow)."},
    {"id": "24gb", "min_gb": 20, "label": "20-24 GB (e.g. RTX 4090 / 3090)", "supported": True,
     "unets": ["ltx-2.5-22b-distilled-Q6_K.gguf", "ltx-2.5-22b-distilled-Q5_K_M.gguf",
               "ltx-2.5-22b-distilled-Q4_K_M.gguf"],
     "resolutions": ["640x352", "768x448"], "chunk_frames": 57, "overlap": 9, "max_seconds": 10, "upscale": "yes",
     "upscale_max_frames": 57, "reserve_vram": 3,
     "why": "Room for Q5/Q6 weights and ~2x the tokens per chunk; upscale per chunk."},
    {"id": "32gb", "min_gb": 28, "label": "28-32 GB (e.g. RTX 5090)", "supported": True,
     "unets": ["ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
               "ltx-2.5-22b-distilled-Q8_0.gguf", "ltx-2.5-22b-distilled-Q6_K.gguf",
               "ltx-2.5-22b-distilled-Q4_K_M.gguf"],
     "resolutions": ["640x352", "768x448", "960x544"], "chunk_frames": 97, "overlap": 17, "max_seconds": 15,
     "upscale": "yes", "upscale_max_frames": 97, "reserve_vram": 2,
     "why": "int8 weights fit entirely; long chunks at up to 960x544."},
    {"id": "48gb", "min_gb": 40, "label": "40-80 GB (RTX 6000 / A100 / H100, hosted)", "supported": True,
     "unets": ["ltx-2.5-22b-distilled-transformer-bf16.safetensors",
               "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
               "ltx-2.5-22b-distilled-Q8_0.gguf", "ltx-2.5-22b-distilled-Q4_K_M.gguf"],
     "resolutions": ["640x352", "768x448", "960x544", "1280x704"], "chunk_frames": 121, "overlap": 17,
     "max_seconds": 20, "upscale": "yes", "upscale_max_frames": 121, "reserve_vram": 2,
     "why": "bf16 (80 GB) or int8 (48 GB) weights; up to 1280x704 and 5 s chunks."},
]


def tier_for_vram(total_mb: int) -> dict:
    gb = total_mb / GB
    t = TIERS[0]
    for cand in TIERS:
        if gb + 0.5 >= cand["min_gb"]:      # 16 GB cards report ~15.9 GB
            t = cand
    return t


def tier_by_id(tid: str) -> dict:
    return next((t for t in TIERS if t["id"] == tid), TIERS[1])


def _get(base: str, path: str, timeout: float = 8.0):
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as r:
        return json.loads(r.read())


def nvidia_smi() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,memory.used,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8).stdout
        name, tot, used, drv = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"name": name, "total_mb": int(tot), "used_mb": int(used), "driver": drv}
    except Exception:  # noqa: BLE001
        return None


def reserve_vram(argv: list[str]) -> float:
    for i, a in enumerate(argv):
        if a == "--reserve-vram" and i + 1 < len(argv):
            try:
                return float(argv[i + 1])
            except ValueError:
                return 0.0
    return 0.0


def installed_files(base: str) -> dict[tuple, list[str]]:
    want = {UNET_GGUF, UNET_ST} | {m["loader"] for m in SHARED_MODELS}
    found: dict[tuple, list[str]] = {}
    for node, field in want:
        try:
            spec = _get(base, f"/object_info/{node}")[node]["input"]["required"][field]
            opts = spec[0] if isinstance(spec[0], list) else (spec[1] or {}).get("options", [])  # legacy list | COMBO
            found[(node, field)] = list(opts)
        except Exception:  # noqa: BLE001
            found[(node, field)] = []
    return found


def detect(base_url: str | None = None, hosted: bool | None = None, use_smi: bool | None = None,
           files: dict | None = None) -> dict:
    base_url = base_url or os.environ.get("COMFY_LOCAL_URL", "http://127.0.0.1:8001")
    hosted = (os.environ.get("RESHOOT_BACKEND", "mock") in ("comfy", "proxy")) if hosted is None else hosted
    local = any(h in base_url for h in ("127.0.0.1", "localhost"))
    use_smi = local if use_smi is None else use_smi
    rep: dict = {"comfy_url": base_url, "reachable": False, "hosted": hosted, "warnings": []}
    stats = None
    try:
        stats = _get(base_url, "/system_stats")
        rep["reachable"] = True
    except Exception as e:  # noqa: BLE001
        rep["warnings"].append(f"Could not reach ComfyUI at {base_url}: {e}")
    sysd = (stats or {}).get("system", {})
    dev = ((stats or {}).get("devices") or [{}])[0]
    argv = sysd.get("argv") or []
    rep.update({
        "gpu": (dev.get("name") or "").split(" : ")[0].replace("cuda:0 ", "") or None,
        "vram_total_mb": int(dev.get("vram_total", 0)) // (1 << 20),
        "vram_free_mb": int(dev.get("vram_free", 0)) // (1 << 20),
        "torch": sysd.get("pytorch_version"), "comfyui": sysd.get("comfyui_version"),
        "ram_total_gb": round(int(sysd.get("ram_total", 0)) / (1 << 30), 1),
        "ram_free_gb": round(int(sysd.get("ram_free", 0)) / (1 << 30), 1),
        "launch_args": argv, "reserve_vram": reserve_vram(argv),
        "lowvram": "--lowvram" in argv or "--novram" in argv,
    })
    rep["cuda"] = (rep["torch"] or "").split("+cu")[-1] if "+cu" in (rep["torch"] or "") else None
    smi = nvidia_smi() if use_smi else None
    rep["nvidia_smi"] = smi
    if smi and not rep["gpu"]:
        rep["gpu"], rep["vram_total_mb"] = smi["name"], smi["total_mb"]
    if hosted and not rep["vram_total_mb"]:
        tier = dict(tier_by_id(os.environ.get("RESHOOT_HOSTED_TIER", "48gb")))
        tier["why"] = "Hosted ComfyUI: hardware not reported, assuming RESHOOT_HOSTED_TIER=" + tier["id"] + ". " + tier["why"]
    else:
        tier = dict(tier_for_vram(rep["vram_total_mb"] or (smi or {}).get("total_mb", 0)))
    forced = os.environ.get("RESHOOT_TIER")
    if forced:
        tier = dict(tier_by_id(forced)); tier["why"] = f"Forced by RESHOOT_TIER={forced}. " + tier["why"]
    rep["tier"] = tier

    # models present / missing
    inst = files if files is not None else (installed_files(base_url) if rep["reachable"] else {})
    models = []
    for m in SHARED_MODELS:
        have = m["file"] in inst.get(m["loader"], [])
        models.append({"role": m["role"], "file": m["file"], "folder": m["folder"], "size_gb": m["gb"],
                       "present": have, "required": m["required"] or (m["role"].startswith("2x") and tier["upscale"] != "no"),
                       "url": m["url"]})
    unet = None
    for f in tier["unets"]:
        loader, gb_, url = UNETS[f]
        have = f in inst.get(loader, [])
        if have and not unet:
            unet = f
        models.append({"role": "LTX-2.5 transformer" + (" (preferred)" if f == tier["unets"][0] else " (fallback)"),
                       "file": f, "folder": "unet" if loader == UNET_GGUF else "diffusion_models", "size_gb": gb_,
                       "present": have, "required": f == tier["unets"][0] and not unet, "url": url})
    rep["models"] = models
    rep["unet"] = unet
    rep["unet_loader"] = (UNETS[unet][0][0] if unet else None)
    missing = [m for m in models if m["required"] and not m["present"]]
    if rep["reachable"] and not hosted:
        if not unet and tier["supported"]:
            rep["warnings"].append("No LTX-2.5 transformer for this tier is installed (see the download list).")
        if rep["reserve_vram"] < tier["reserve_vram"] and not rep["lowvram"]:
            rep["warnings"].append(f"Start ComfyUI with --reserve-vram {tier['reserve_vram']:g} (currently "
                                   f"{rep['reserve_vram']:g}). Without it the GPU can fill up and reset the display driver.")
    rep["missing"] = [m["file"] for m in missing if not m["role"].startswith("LTX-2.5 transformer")]
    rep["can_render"] = bool(rep["reachable"] and tier["supported"] and (unet or hosted) and not rep["missing"]
                             and (hosted or rep["lowvram"] or rep["reserve_vram"] >= min(tier["reserve_vram"], 2)))
    rep["caps"] = caps(tier)
    return rep


def caps(tier: dict) -> dict:
    """What the UI may offer on this tier."""
    return {"resolutions": tier["resolutions"],
            "resolutions_by_aspect": {a: tier_resolutions(tier, a) for a in ASPECTS},
            "max_seconds": tier["max_seconds"],
            "chunk_frames": tier["chunk_frames"], "overlap": tier["overlap"],
            "upscale": tier["upscale"], "upscale_max_frames": tier["upscale_max_frames"],
            "can": [s for s in [
                f"Re-shoot up to {tier['max_seconds']} s" if tier["supported"] else None,
                f"Resolution up to {tier['resolutions'][-1]}" if tier["resolutions"] else None,
                (f"{tier['chunk_frames']} frames per chunk, {tier['overlap']} frame overlap") if tier["supported"] else None,
                {"yes": "2x latent upscale", "short": f"2x upscale on clips up to {tier['upscale_max_frames']} frames (slow)",
                 "no": None}[tier["upscale"]],
            ] if s],
            "cannot": [s for s in [
                None if tier["supported"] else "Render locally (GPU too small)",
                (f"Clips longer than {tier['max_seconds']} s") if tier["supported"] else None,
                ("Resolutions above " + tier["resolutions"][-1]) if tier["resolutions"] and tier["resolutions"][-1] != "1280x704" else None,
                {"short": f"Upscale on clips longer than {tier['upscale_max_frames']} frames", "no": "2x upscale",
                 "yes": None}[tier["upscale"]],
            ] if s]}


def summary_text(rep: dict) -> str:
    t = rep["tier"]
    lines = [f"ComfyUI: {rep['comfy_url']} ({'reachable' if rep['reachable'] else 'NOT reachable'})",
             f"GPU: {rep.get('gpu')}  VRAM {rep['vram_total_mb']} MB total / {rep['vram_free_mb']} MB free",
             f"torch {rep.get('torch')}  CUDA {rep.get('cuda')}  RAM {rep['ram_total_gb']} GB  "
             f"--reserve-vram {rep['reserve_vram']:g}",
             f"Tier: {t['label']} -> {t['why']}"]
    if rep.get("nvidia_smi"):
        s = rep["nvidia_smi"]
        lines.append(f"nvidia-smi: {s['name']} driver {s['driver']}, {s['used_mb']}/{s['total_mb']} MB used")
    lines += ["Can: " + "; ".join(rep["caps"]["can"]), "Cannot: " + ("; ".join(rep["caps"]["cannot"]) or "-"),
              f"Model: {rep['unet'] or 'none installed'}", "Models:"]
    for m in rep["models"]:
        mark = "OK " if m["present"] else ("MISSING" if m["required"] else "optional")
        lines.append(f"  [{mark:8}] {m['file']} ({m['size_gb']} GB, models/{m['folder']}/)"
                     + ("" if m["present"] else f"  <- {m['url']}"))
    lines += [f"WARNING: {w}" for w in rep["warnings"]]
    lines.append("READY to render" if rep["can_render"] else "NOT ready to render")
    return "\n".join(lines)


# ---- aspect-ratio-aware output -------------------------------------------------------------
# LTX latents are 32 px per side (EmptyLTXVLatentVideo width/height step=32), so every output dim is a
# multiple of 32. Each tier lists its landscape sizes; square/portrait variants keep ~the same pixel count
# (VRAM is ~linear in width*height*frames) and never exceed it.
ASPECTS = {"landscape": 16 / 9, "square": 1.0, "portrait": 9 / 16}


def aspect_of(width: int, height: int) -> str:
    r = width / max(height, 1)
    if r > 1.2:
        return "landscape"
    if r < 1 / 1.2:
        return "portrait"
    return "square"


def res_for_aspect(base: str, aspect: str) -> str:
    """Tier-equivalent size for an aspect: same pixel budget as `base` (e.g. 640x352), dims % 32 == 0."""
    bw, bh = (int(x) for x in base.split("x"))
    if aspect == "landscape":
        return base
    if aspect == "portrait":
        return f"{bh}x{bw}"
    budget = bw * bh          # never above the landscape budget (480x480 tripped the 16 GB watchdog)
    s = int((budget ** 0.5) // 32) * 32
    return f"{s}x{s}"


def tier_resolutions(tier: dict, aspect: str | None = None) -> list[str]:
    """All allowed sizes for a tier (all aspects), or just those of one aspect."""
    asp = [aspect] if aspect else list(ASPECTS)
    return [res_for_aspect(r, a) for a in asp for r in tier["resolutions"]]
