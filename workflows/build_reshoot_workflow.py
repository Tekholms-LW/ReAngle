"""Generate the ReAngle LTX-2.5 API-format workflow (ComfyUI /prompt format).

Pipeline: VHS_LoadVideo -> Video Depth Anything -> ReAngle camera path -> ReAngle depth reproject
-> LTX-2.5 (GGUF) + IC-LoRA guide (warped RGB with Layout-to-Render, or warped depth with Union Control)
   + first frame I2V anchor (path starts at identity, so frame 0 == original frame 0)
-> distilled 8-step sampling -> crop guides -> [optional 2x latent upscale + 3-step refine]
-> tiled decode -> VHS_VideoCombine (h264) with the ORIGINAL audio.

Usage: python build_reshoot_workflow.py [--guide layout|depth] [--upscale] [-o out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = {
    "unet": "ltx-2.5-22b-distilled-Q4_K_M.gguf",
    "text_encoder": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "video_vae": "ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": "ltx-2.5-audio-vae-bf16.safetensors",
    "upscaler": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "lora_layout": "ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors",
    "lora_depth": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    "vda": "video_depth_anything_vits.pth",
}
SIGMAS_S1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
SIGMAS_S2 = "0.909375, 0.725, 0.421875, 0.0"


def build(guide: str = "layout", upscale: bool = False, width: int = 640, height: int = 352,
          frames: int = 41, fps: int = 24, unet: str | None = None, chunk: bool = False,
          prev_frames: int = 0, fill: bool = False, anchor: float = 0.0, prev_strength: float = 1.0) -> dict:
    """chunk=True: the warped guide is precomputed by the app for the whole clip (one depth pass, one global camera
    path) and loaded from `guide.mp4`; `prev_frames` > 0 conditions the start of this chunk on the last frames of
    the previous chunk's output (LTXVAddGuide at frame 0) instead of the source first frame (I2V anchor)."""
    g: dict = {}

    def n(i, cls, title, **inputs):
        g[str(i)] = {"class_type": cls, "inputs": inputs, "_meta": {"title": title}}
        return str(i)

    L = lambda i, s=0: [str(i), s]  # noqa: E731
    # --- inputs
    n(1, "VHS_LoadVideo", "Source clip", video="input.mp4", force_rate=fps, custom_width=width,
      custom_height=height, frame_load_cap=frames, skip_first_frames=0, select_every_nth=1, format="LTXV")
    n(2, "ImageFromBatch", "First frame (anchor)", image=L(1), batch_index=0, length=1)
    # --- depth + reprojection
    if chunk:
        n(23, "VHS_LoadVideo", "Warped guide (precomputed)", video="guide.mp4", force_rate=fps, custom_width=width,
          custom_height=height, frame_load_cap=frames, skip_first_frames=0, select_every_nth=1, format="LTXV")
    else:
        _depth_nodes(n, L, frames)
    unet = unet or MODELS["unet"]
    if unet.endswith(".gguf"):
        n(30, "UnetLoaderGGUF", "LTX-2.5 " + unet, unet_name=unet)
    else:
        n(30, "UNETLoader", "LTX-2.5 " + unet, unet_name=unet, weight_dtype="default")
    if chunk and fill:   # regen mask: white = revealed / out-of-frame area the model should generate itself
        n(25, "VHS_LoadVideo", "Regen mask (precomputed)", video="mask.mp4", force_rate=fps, custom_width=width,
          custom_height=height, frame_load_cap=frames, skip_first_frames=0, select_every_nth=1, format="LTXV")
        n(26, "ImageToMask", "Regen mask -> MASK", image=L(25), channel="red")
        n(27, "InvertMask", "Guide attention (1 = keep warped pixels)", mask=L(26))
    _rest(n, L, g, guide, upscale, width, height, frames, fps, chunk, prev_frames, chunk and fill)
    if chunk and prev_frames and anchor > 0:
        # re-anchor later chunks to the SOURCE: their first frame is the warped original (from the guide), not only
        # the previous chunk's generated tail, so appearance drift can't accumulate chunk over chunk
        g["2"]["inputs"]["image"] = L(23)
        g["2"]["_meta"]["title"] = "Chunk anchor (warped source frame)"
        g["51"]["inputs"].update(bypass=False, strength=float(anchor))
        g["55"]["inputs"]["strength"] = float(prev_strength)
    return g


def _depth_nodes(n, L, frames):
    n(10, "LoadVideoDepthAnythingModel", "VDA model", model=MODELS["vda"])
    n(11, "VideoDepthAnythingProcess", "Depth", vda_model=L(10), images=L(1), input_size=518, max_res=960, precision="fp16")
    n(12, "VideoDepthAnythingOutput", "Depth (gray)", depths=L(11), colormap="gray")
    n(20, "ReAngleCameraPathFromPreset", "Camera path", preset="orbit_left", angle_deg=20.0,
      num_frames=frames, pivot_depth=3.0, hold=False)
    n(22, "ReAngleCameraPathFromJSON", "Custom camera path (3D editor)",
      path_json=json.dumps({"version": 1, "pivot_depth": 3.0, "keyframes": [
          {"t": 0, "pos": [0, 0, 0], "look": [0, 0, 3]}, {"t": 1, "pos": [-0.6, 0, 0.1], "look": [0, 0, 3]}]}),
      num_frames=frames)
    n(21, "ReAngleDepthReproject", "Depth reproject", images=L(1), depth=L(12), camera_path=L(20),
      hfov_deg=60.0, near=1.0, far=10.0, inverse_depth=True, crack_fill_passes=2)


def _rest(n, L, g, guide, upscale, width, height, frames, fps, chunk, prev_frames, fill=False):
    # --- models
    n(31, "CLIPLoader", "Gemma text encoder", clip_name=MODELS["text_encoder"], type="ltxv", device="default")
    n(32, "VAELoader", "Video VAE", vae_name=MODELS["video_vae"])
    n(33, "VAELoader", "Audio VAE", vae_name=MODELS["audio_vae"])
    lora = MODELS["lora_layout"] if guide == "layout" else MODELS["lora_depth"]
    n(34, "LTXICLoRALoaderModelOnly", "IC-LoRA", model=L(30), lora_name=lora, strength_model=1.0)
    # --- conditioning
    n(40, "CLIPTextEncode", "Positive", clip=L(31),
      text="The same scene filmed from a new camera angle, consistent lighting, detailed, cinematic, smooth camera motion. "
           "Only the original subject; the revealed background continues the same empty, out-of-focus backdrop.")
    # NB: the distilled model samples at cfg=1.0 (node 60), where ComfyUI skips the negative pass, so this text only
    # takes effect if cfg is raised; the positive-prompt clause above is what steers the background at cfg=1.
    n(41, "CLIPTextEncode", "Negative", clip=L(31),
      text="person, figure, silhouette, extra people in background, blurry, distorted, warped, flicker, artifacts, "
           "text, watermark")
    n(42, "LTXVConditioning", "fps conditioning", positive=L(40), negative=L(41), frame_rate=float(fps))
    # --- latents
    n(50, "EmptyLTXVLatentVideo", "Empty video latent", width=width, height=height, length=frames, batch_size=1)
    n(51, "LTXVImgToVideoInplace", "I2V anchor", vae=L(32), image=L(2), latent=L(50), strength=1.0,
      bypass=bool(prev_frames))
    guide_img = L(23, 0) if chunk else (L(21, 0) if guide == "layout" else L(21, 2))
    n(52, "LTXAddVideoICLoRAGuide", "IC-LoRA guide", positive=L(42, 0), negative=L(42, 1), vae=L(32), latent=L(51),
      image=guide_img, frame_idx=0, strength=1.0, latent_downscale_factor=L(34, 1), crop="disabled",
      use_tiled_encode=False, tile_size=256, tile_overlap=64)
    if fill:     # same guide, but its attention is switched off where the warp revealed nothing (holes, borders)
        g["52"]["class_type"] = "LTXAddVideoICLoRAGuideAdvanced"
        g["52"]["inputs"].update(attention_strength=1.0, attention_mask=L(27))
        g["52"]["_meta"]["title"] = "IC-LoRA guide (masked: model fills revealed areas)"
    cond = 52
    if prev_frames:
        n(24, "VHS_LoadVideo", "Previous chunk tail", video="prev.mp4", force_rate=fps, custom_width=width,
          custom_height=height, frame_load_cap=prev_frames, skip_first_frames=0, select_every_nth=1, format="LTXV")
        n(55, "LTXVAddGuide", "Continue from previous chunk", positive=L(52, 0), negative=L(52, 1), vae=L(32),
          latent=L(52, 2), image=L(24), frame_idx=0, strength=1.0)
        cond = 55
    n(53, "LTXVEmptyLatentAudio", "Empty audio latent", frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=L(33))
    n(54, "LTXVConcatAVLatent", "AV latent", video_latent=L(cond, 2), audio_latent=L(53))
    # --- stage 1
    n(60, "CFGGuider", "Guider", model=L(34), positive=L(cond, 0), negative=L(cond, 1), cfg=1.0)
    n(61, "RandomNoise", "Noise", noise_seed=42)
    n(62, "KSamplerSelect", "Sampler", sampler_name="euler_ancestral")
    n(63, "ManualSigmas", "Sigmas (8 steps)", sigmas=SIGMAS_S1)
    n(64, "SamplerCustomAdvanced", "Stage 1", noise=L(61), guider=L(60), sampler=L(62), sigmas=L(63), latent_image=L(54))
    n(65, "LTXVSeparateAVLatent", "Split AV", av_latent=L(64))
    n(66, "LTXVCropGuides", "Crop guides", positive=L(cond, 0), negative=L(cond, 1), latent=L(65, 0))
    video_latent = L(66, 2)
    if upscale:
        n(70, "LatentUpscaleModelLoader", "Upscaler", model_name=MODELS["upscaler"])
        n(71, "LTXVLatentUpsampler", "2x latent upscale", samples=L(66, 2), upscale_model=L(70), vae=L(32))
        n(72, "LTXVConcatAVLatent", "AV latent 2", video_latent=L(71), audio_latent=L(65, 1))
        n(73, "CFGGuider", "Guider 2", model=L(30), positive=L(66, 0), negative=L(66, 1), cfg=1.0)
        n(74, "ManualSigmas", "Sigmas (3 steps)", sigmas=SIGMAS_S2)
        n(75, "KSamplerSelect", "Sampler 2", sampler_name="euler")
        n(76, "SamplerCustomAdvanced", "Stage 2", noise=L(61), guider=L(73), sampler=L(75), sigmas=L(74), latent_image=L(72))
        n(77, "LTXVSeparateAVLatent", "Split AV 2", av_latent=L(76))
        video_latent = L(77, 0)
    # --- decode + save with original audio
    n(80, "VAEDecodeTiled", "Decode", samples=video_latent, vae=L(32), tile_size=512, overlap=64,
      temporal_size=64, temporal_overlap=8)
    n(90, "VHS_VideoCombine", "Save MP4", images=L(80), frame_rate=float(fps), loop_count=0,
      filename_prefix="reangle/reshoot", format="video/h264-mp4", pingpong=False, save_output=True,
      audio=L(1, 2), pix_fmt="yuv420p", crf=19, save_metadata=True, trim_to_audio=False)
    if chunk:      # the app crossfades chunks and muxes the original audio itself
        del g["90"]["inputs"]["audio"]
        g["90"]["inputs"]["filename_prefix"] = "reangle/chunk"
        return
    n(91, "VHS_VideoCombine", "Save warped preview", images=L(21, 0), frame_rate=float(fps), loop_count=0,
      filename_prefix="reangle/warped", format="video/h264-mp4", pingpong=False, save_output=True,
      pix_fmt="yuv420p", crf=19, save_metadata=False, trim_to_audio=False)


BINDINGS = {
    "video": ["1", "video"], "frames": ["1", "frame_load_cap"], "width": ["1", "custom_width"],
    "height": ["1", "custom_height"], "preset": ["20", "preset"], "angle_deg": ["20", "angle_deg"],
    "num_frames": ["20", "num_frames"], "path_json": ["22", "path_json"], "num_frames_json": ["22", "num_frames"],
    "camera_input": ["21", "camera_path"], "prompt": ["40", "text"], "seed": ["61", "noise_seed"],
    "output_node": "90",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--guide", choices=["layout", "depth"], default="layout")
    ap.add_argument("--upscale", action="store_true")
    ap.add_argument("-o", default=str(Path(__file__).with_name("ltx25_reshoot_api.json")))
    a = ap.parse_args()
    wf = build(a.guide, a.upscale)
    # node 22 is only used when the app sends a custom path (it rewires 21.camera_path and drops node 20)
    out = {"_comment": f"ReAngle LTX-2.5 re-shoot graph (guide={a.guide}, upscale={a.upscale}). Generated by build_reshoot_workflow.py. "
           "Models: " + ", ".join(MODELS.values()),
           "_reshoot_bindings": BINDINGS, **wf}
    Path(a.o).write_text(json.dumps(out, indent=1))
    print("wrote", a.o, len(wf), "nodes")
