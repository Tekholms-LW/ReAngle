# ReAngle

**Re-shoot any video from a new camera angle, locally, with open weights.**

![Original (left) vs. ReAngle 12° orbit (right)](docs/media/hero.gif)

*Left: the source clip. Right: the same clip re-shot with a 12° orbit, rendered on a 16 GB RTX 5080. More: [before/after mp4](docs/media/before_after_orbit12.mp4) · [9:16 stacked](docs/media/social_stacked.mp4) · [9:16 wipe](docs/media/social_wipe.mp4).*

ReAngle is a clean-room, open-source take on "re-shoot a video": upload a clip, pick a camera move or draw your own in 3D, and get the same performance back from a new viewpoint. It runs on ComfyUI with **LTX-2.5** as the core video model. Built for Comfy's [Open Source Higgsfield](https://comfy.org/open-source-higgsfield) challenge.

## What's new here

- **Draw your own camera path in the clip's own 3D space.** The first frame is lifted into a depth mesh (Video Depth Anything); you drag the camera, add keyframes and scrub a live warped preview. Presets (orbit, low/high, push/pull, truck, crane, "matrix" freeze-time) are just editable keyframe seeds. A safety envelope (±30° turn, limited travel) is drawn while you edit.
- **Depth reprojection → LTX IC-LoRA guide.** The whole clip is depth-estimated once, warped along one global camera path (z-buffered splat), and fed to LTX-2.5 through the Layout-to-Render IC-LoRA.
- **Background-aware fill of revealed areas.** Holes opened by the new viewpoint are pre-filled from the *far* side (never from the subject, so no ghost silhouettes), frame borders are mirrored from background only, see-through specks in hair are removed, and a feathered mask switches the guide's attention off where the warp has no real pixels, so LTX paints those areas itself.
- **Long clips without drift.** Overlapping chunks are crossfaded; each later chunk is re-anchored to the warped *source* frame (not only the previous output) and colour-matched in LAB to the real pixels. Drift is measured per chunk with `scripts/drift_metrics.py`.
- **Safe on a 16 GB card.** Hardware detection picks a tier (resolution, chunk length, upscale), refuses under-spec GPUs, checks ComfyUI's `--reserve-vram`, requires free VRAM and an empty queue before each chunk, runs one job at a time, and a ~5 Hz `nvidia-smi` watchdog interrupts ComfyUI before the display driver can reset.
- **Aspect-aware output.** Landscape, square and portrait clips render at a tier-equivalent pixel budget (dims multiple of 32): e.g. 640×352 / 448×448 / 352×640 on 16 GB. No letterbox crop of the subject.
- **One-click 9:16 social export.** Stacked before/after or an animated wipe, 1080×1920 h264 + AAC, faststart, blurred-fill background for any aspect, optional credit line. CPU only.

## How it works

```
clip ─ffmpeg trim/scale─► Video Depth Anything (whole clip) ─► camera path (preset or drawn)
     ─► depth reprojection (z-buffer) ─► background-aware prefill + regen mask
     ─► LTX-2.5 distilled (GGUF) + Layout-to-Render IC-LoRA guide (attention-masked)
        + first-frame / re-anchor conditioning, chunked with overlap
     ─► crossfade + LAB colour match ─► original audio muxed back ─► mp4 (+ optional 9:16 export)
```

| Folder | What |
|---|---|
| `backend/` | FastAPI app: uploads, job queue, hardware tiers, ComfyUI client (local native API or hosted Comfy API), chunked renderer, social export |
| `frontend/` | Vite + TypeScript + three.js UI: upload, presets, 3D path editor, hardware panel, progress, before/after wipe, social export |
| `comfyui-reshoot-nodes/` | ComfyUI node pack **comfyui-reangle**: camera paths (preset/JSON) and depth reprojection (numpy) |
| `workflows/` | API-format LTX-2.5 graphs + `build_reshoot_workflow.py` that generates them |
| `scripts/` | `check_hardware.py`, `comfy_run.py` (run a graph headless), `drift_metrics.py`, setup/run/test |

## Open weights used

| Model | File(s) (ComfyUI folder) | Source | License |
|---|---|---|---|
| LTX-2.5 22B distilled, Q4_K_M GGUF (16 GB tier) | `ltx-2.5-22b-distilled-Q4_K_M.gguf` (`models/unet/`) | [unsloth/LTX-2.5-GGUF](https://huggingface.co/unsloth/LTX-2.5-GGUF) (quant of [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5)) | [LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x): free incl. commercial for entities under $10M annual revenue; larger entities need a paid license |
| LTX-2.5 video + audio VAE, latent upscaler ×2 | `ltx-2.5-video-vae-bf16`, `ltx-2.5-audio-vae-bf16` (`models/vae/`), `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0` (`models/latent_upscale_models/`) | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | LTX-2.x Community License |
| Gemma 4 12B text encoder (LTX packaging, int8) | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` (`models/text_encoders/`) | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | Gemma 4 is Apache-2.0; the LTX projection packaging ships in the LTX repo (treat as LTX-2.x Community License) |
| IC-LoRA Layout-to-Render (guide, default) | `ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors` (`models/loras/`) | [Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render) | LTX-2.x Community License |
| IC-LoRA Union Control (optional depth guide) | `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` (`models/loras/`) | [Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control) | LTX-2.x Community License |
| Video Depth Anything **Small** | `video_depth_anything_vits.pth` (`models/videodepthanything/`) | [depth-anything/Video-Depth-Anything-Small](https://huggingface.co/depth-anything/Video-Depth-Anything-Small) | Apache-2.0 |

> ⚠️ Use the **Small** depth model. Video Depth Anything **Base/Large** are **CC-BY-NC-4.0 (non-commercial)**; ReAngle defaults to Small for that reason.
> Higher VRAM tiers can use other LTX-2.5 transformers (Q5/Q6/Q8 GGUF, int8, bf16); all fall under the LTX-2.x Community License.
> No model weights are included in this repo. No partner nodes or paid APIs are needed for local use.

## Requirements

- NVIDIA GPU, **12–16 GB VRAM minimum** (tested: RTX 5080 16 GB on Windows 11). Under 12 GB is refused (use the hosted backend).
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (Desktop or portable) with these custom nodes:
  - [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) (IC-LoRA guide nodes)
  - [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (`UnetLoaderGGUF`)
  - [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) (load/save video)
  - [ComfyUI-Video-Depth-Anything](https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything) (depth)
  - **comfyui-reangle** (this repo's `comfyui-reshoot-nodes/`)
- Python 3.10+, Node 18+, ffmpeg on PATH.

## Install

```bash
git clone https://github.com/Tekholms-LW/ReAngle && cd ReAngle

# 1. ComfyUI nodes: copy (or symlink) the pack into ComfyUI
cp -r comfyui-reshoot-nodes /path/to/ComfyUI/custom_nodes/comfyui-reangle
#    and install the four custom-node packs above (ComfyUI Manager works)

# 2. Models: download the files in the table above into the listed ComfyUI model folders

# 3. Start ComfyUI with VRAM headroom for the desktop (required on 16 GB cards)
#    e.g. python main.py --port 8001 --reserve-vram 3
#    Comfy Desktop: add "--port 8001 --reserve-vram 3" to the launch args and restart

# 4. The app
make setup                 # venv + backend deps + UI build   (or scripts/setup.sh)
python scripts/check_hardware.py   # ✓/missing for every model, your tier, what you can render
RESHOOT_BACKEND=local COMFY_LOCAL_URL=http://127.0.0.1:8001 make run   # http://127.0.0.1:8000
```

Windows (PowerShell): create a venv in the repo, `pip install -r backend/requirements.txt`, `cd frontend; npm ci; npm run build`, then
`$env:RESHOOT_BACKEND="local"; .venv\Scripts\uvicorn reshoot.main:app --app-dir backend --port 8000`.

No GPU? `make run` uses the **mock** backend (a 2D stand-in) so you can try the UI. No GPU at all? Run it on Comfy Cloud, see below.

## Use

1. Drop a clip (≤120 s source; the tier decides how much is rendered: 5 s on 16 GB).
2. Pick presets and strength, **or** click **Draw your own path (3D)** and frame the end shot in the depth preview.
3. Choose length and resolution (the one matching your clip's aspect is preselected), keep **Fill revealed areas** on, click **Re-shoot**.
4. Compare with the wipe slider, download, or **Export for social** (stacked or wipe, 1080×1920).

Headless: `python scripts/comfy_run.py workflows/ltx25_reshoot_api.json --set video=clip.mp4 --set angle_deg=12`.

## Run on Comfy Cloud

ReAngle runs on the Comfy Developer Platform with no local GPU. You need a Comfy Cloud plan and an API key from platform.comfy.org.

```bash
export COMFY_API_KEY=...            # keep it out of git; .env is gitignored
RESHOOT_BACKEND=comfy make run      # COMFY_BASE_URL defaults to https://cloud.comfy.org
```

How the work is split. Comfy Cloud has no custom nodes, so ReAngle sends only stock-node graphs:

| Step | Where |
|---|---|
| Depth (Depth Anything V2 Small, per frame) | Comfy Cloud |
| Camera path, depth warp, background-aware fill, regen mask | the ReAngle app (CPU, same code as the node pack) |
| LTX-2.5 22B distilled, **full bf16**, IC-LoRA guide + attention mask, re-anchor | Comfy Cloud |
| Crossfade between chunks, LAB colour match, 9:16 social export | the ReAngle app |

The client uses Cloud's ComfyUI-compatible API: `POST /api/upload/image`, `POST /api/prompt`, `GET /api/jobs/{id}`, `GET /api/view`, and auth through the `X-API-Key` header. A 402 response means you are out of credits. The app logs your credit balance before and after each job.

Measured on a 33-frame 448×448 orbit (12°) of the sample clip: about 50–95 s wall time and $0.03–0.07 of credits per run.

Known difference from local: Comfy Cloud does not host the LTX-2.5 Layout-to-Render IC-LoRA, so Cloud mode uses `ltx-2.3-22b-ic-lora-union-control-ref0.5` (set it with `RESHOOT_CLOUD_LORA`). Results are close to local, with slightly more drift at the frame border.

## Hardware tiers

| Tier | Transformer | Resolutions (landscape / square / portrait) | Frames per chunk (overlap) | Max length | Status |
|---|---|---|---|---|---|
| < 12 GB | – | – | – | – | refused |
| 12–16 GB | Q4_K_M GGUF | 640×352 / 448×448 / 352×640 | 33 (9) | 5 s | **tested** (RTX 5080, peak ≈15.3 of 16.3 GB, ~55 s per chunk) |
| 20–24 GB | Q6_K → Q5_K_M → Q4_K_M | + 768×448 / 576×576 / 448×768 | 57 (9) | 10 s | untested estimate |
| 28–32 GB | int8 → Q8_0 → Q6_K | + 960×544 / 704×704 / 544×960 | 97 (17) | 15 s | untested estimate |
| 40–80 GB / hosted | bf16 → int8 → Q8_0 | + 1280×704 / 928×928 / 704×1280 | 121 (17) | 20 s | untested estimate |

## Limitations (honest)

- Small moves work best (≈10–20° orbits). The model redraws revealed areas, so big moves invent more.
- Faces drift slightly over long clips (face SSIM vs. the warped source ≈0.98 in chunk 1 → ≈0.90 by chunk 4 on our test).
- Thin structures (loose hair, cables), screens/reflections and tiny text are hard for depth reprojection.
- Stylised/anime footage gets flat depth from a photo-trained depth model.
- At 16 GB, output is small (≈0.2 MP) and ≤5 s; use the 2× upscale for short clips.
- The distilled model samples at cfg 1.0, where ComfyUI skips the negative prompt; steering is done with the positive prompt and the masks.
- Only the 16 GB tier is measured.

## Clean-room statement

ReAngle was written from scratch from public behaviour descriptions of "re-shoot"-style apps. It contains **no Higgsfield code, assets, prompts, models or UI**, and no code copied from other re-shoot projects. Third-party models and ComfyUI node packs are used as dependencies under their own licenses (above).

## Tests

```bash
make test        # backend (FastAPI, chunking, fill, watchdog, social export) + node pack
```

## License

Apache-2.0 (see `LICENSE`). Model weights are not part of this repo and keep their own licenses.

## Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and the Comfy Developer Platform
- [Lightricks LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5), [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) and the LTX IC-LoRAs
- [unsloth](https://huggingface.co/unsloth/LTX-2.5-GGUF) GGUF quants, [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) and its ComfyUI wrapper
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite), [three.js](https://threejs.org), [FastAPI](https://fastapi.tiangolo.com), [ffmpeg](https://ffmpeg.org)
- Demo clip: AI-generated with Grok Imagine by the author (no real person depicted).
