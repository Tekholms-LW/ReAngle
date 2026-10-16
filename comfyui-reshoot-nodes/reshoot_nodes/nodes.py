"""ComfyUI node wrappers. torch is imported lazily so the core stays CPU/numpy-testable."""
from __future__ import annotations

import numpy as np

from .camera import MAX_ANGLE_DEG, PRESETS, camera_path, intrinsics
from .camera_json import path_from_json
from .reproject import depth_from_relative, warp_video


class CameraPathFromPreset:
    CATEGORY = "ReAngle"
    RETURN_TYPES = ("RESHOOT_CAMERA_PATH",)
    RETURN_NAMES = ("camera_path",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "preset": (list(PRESETS),),
            "angle_deg": ("FLOAT", {"default": 15.0, "min": -MAX_ANGLE_DEG, "max": MAX_ANGLE_DEG, "step": 1.0}),
            "num_frames": ("INT", {"default": 97, "min": 1, "max": 1024}),
            "pivot_depth": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 100.0, "step": 0.1}),
            "hold": ("BOOLEAN", {"default": False}),
        }}

    def build(self, preset, angle_deg, num_frames, pivot_depth, hold):
        return ({"poses": camera_path(preset, angle_deg, num_frames, pivot_depth, hold=hold)},)


class CameraPathFromJSON:
    """Draw-your-own camera path: keyframes (pos + look-at) from the ReAngle editor -> camera path."""
    CATEGORY = "ReAngle"
    RETURN_TYPES = ("RESHOOT_CAMERA_PATH", "INT")
    RETURN_NAMES = ("camera_path", "clamped_frames")
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "path_json": ("STRING", {"multiline": True, "default":
                '{"version":1,"pivot_depth":3.0,"keyframes":[{"t":0,"pos":[0,0,0],"look":[0,0,3]},'
                '{"t":1,"pos":[-0.8,0,0.2],"look":[0,0,3]}]}'}),
            "num_frames": ("INT", {"default": 97, "min": 1, "max": 1024}),
        }}

    def build(self, path_json, num_frames):
        p = path_from_json(path_json, num_frames)
        return ({"poses": p["poses"], "freeze_time": p["freeze_time"]}, p["clamped_frames"])


class DepthReproject:
    """Warp frames into the new camera path using per-frame depth. Outputs warped frames, hole mask, warped depth."""
    CATEGORY = "ReAngle"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("warped", "hole_mask", "warped_depth")
    FUNCTION = "warp"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "depth": ("IMAGE",),
            "camera_path": ("RESHOOT_CAMERA_PATH",),
            "hfov_deg": ("FLOAT", {"default": 60.0, "min": 10.0, "max": 150.0}),
            "near": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0}),
            "far": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 1000.0}),
            "inverse_depth": ("BOOLEAN", {"default": True}),
        }, "optional": {
            "crack_fill_passes": ("INT", {"default": 2, "min": 0, "max": 8,
                                          "tooltip": "Fill thin splat cracks; big disocclusions stay holes"}),
        }}

    def warp(self, images, depth, camera_path, hfov_deg, near, far, inverse_depth, crack_fill_passes=2):
        import torch
        imgs = images.detach().cpu().numpy().astype(np.float32)           # (N,H,W,C)
        d = depth.detach().cpu().numpy()
        d = d[..., 0] if d.ndim == 4 else d                               # (N,H,W)
        if d.shape[1:] != imgs.shape[1:3]:
            raise ValueError("depth and images must have the same size")
        metric = depth_from_relative(d, near, far, inverse_depth)
        if camera_path.get("freeze_time"):          # "matrix freeze": time stops, camera keeps moving
            n = len(camera_path["poses"])
            imgs = np.repeat(imgs[:1], n, axis=0)
            metric = np.repeat(metric[:1], n, axis=0)
        N, H, W, _ = imgs.shape
        K = intrinsics(W, H, hfov_deg)
        warped, masks, wd = warp_video(imgs, metric, K, camera_path["poses"], crack_passes=crack_fill_passes)
        wd_vis = np.where(wd > 0, (1.0 / np.maximum(wd, 1e-6) - 1.0 / far) / (1.0 / near - 1.0 / far), 0.0)
        wd_img = np.repeat(np.clip(wd_vis, 0, 1)[..., None], 3, axis=-1).astype(np.float32)
        return (torch.from_numpy(warped), torch.from_numpy(masks), torch.from_numpy(wd_img))


NODE_CLASS_MAPPINGS = {"ReAngleCameraPathFromPreset": CameraPathFromPreset, "ReAngleDepthReproject": DepthReproject,
                       "ReAngleCameraPathFromJSON": CameraPathFromJSON}
NODE_DISPLAY_NAME_MAPPINGS = {"ReAngleCameraPathFromPreset": "Camera Path From Preset (ReAngle)",
                              "ReAngleDepthReproject": "Depth Reproject (ReAngle)",
                              "ReAngleCameraPathFromJSON": "Camera Path From JSON (ReAngle)"}
