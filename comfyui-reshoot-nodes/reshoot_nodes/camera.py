"""Camera paths for re-shooting. Pure numpy.

Convention: OpenCV-style camera (x right, y down, z forward). A path is an array of
4x4 matrices T_k mapping points from the SOURCE camera frame k into the NEW camera frame k
(i.e. a relative "source -> target" transform per frame). Identity == original shot.
"""
from __future__ import annotations

import numpy as np

MAX_ANGLE_DEG = 30.0
PRESETS = ("orbit_left", "orbit_right", "low", "high", "push", "pull", "truck", "crane")


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _pivot(R: np.ndarray, pivot_z: float) -> np.ndarray:
    """Rigid transform that rotates the camera around a point pivot_z in front of it."""
    T = np.eye(4)
    p = np.array([0.0, 0.0, pivot_z])
    T[:3, :3] = R
    T[:3, 3] = p - R @ p
    return T


def _trans(t) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = t
    return T


def camera_path(preset: str, angle_deg: float, num_frames: int, pivot_depth: float = 3.0,
                ease: str = "smooth", hold: bool = False) -> np.ndarray:
    """Return (num_frames, 4, 4) source->target transforms.

    angle_deg is clamped to +-MAX_ANGLE_DEG; for translational moves it scales the
    distance (30 deg == ~pivot_depth * 0.5 of travel). hold=True applies the full
    offset on every frame (a static new angle) instead of animating into it.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    a = np.deg2rad(np.clip(angle_deg, -MAX_ANGLE_DEG, MAX_ANGLE_DEG))
    k = abs(a) / np.deg2rad(MAX_ANGLE_DEG)
    sgn = 1.0 if angle_deg >= 0 else -1.0
    t = np.ones(num_frames) if hold else np.linspace(0.0, 1.0, num_frames)
    if ease == "smooth" and not hold:
        t = t * t * (3 - 2 * t)
    travel = 0.5 * pivot_depth * k
    out = np.empty((num_frames, 4, 4))
    for i, u in enumerate(t):
        if preset == "orbit_left":
            # Camera moves left around the subject: scene rotates +y in camera frame.
            T = _pivot(_rot_y(abs(a) * u), pivot_depth)
        elif preset == "orbit_right":
            T = _pivot(_rot_y(-abs(a) * u), pivot_depth)
        elif preset == "low":
            # Camera drops (scene moves up, -y) and tilts up to keep pivot centered.
            T = _pivot(_rot_x(-abs(a) * u), pivot_depth)
        elif preset == "high":
            T = _pivot(_rot_x(abs(a) * u), pivot_depth)
        elif preset == "push":
            T = _trans([0, 0, -travel * u])
        elif preset == "pull":
            T = _trans([0, 0, travel * u])
        elif preset == "truck":
            T = _trans([-sgn * travel * u, 0, 0])
        else:  # crane: camera rises -> scene moves down (+y), slight tilt down
            T = _trans([0, travel * u, 0]) @ _pivot(_rot_x(0.25 * abs(a) * u), pivot_depth)
        out[i] = T
    return out


def intrinsics(width: int, height: int, hfov_deg: float = 60.0) -> np.ndarray:
    f = 0.5 * width / np.tan(np.deg2rad(hfov_deg) / 2)
    return np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1.0]])
