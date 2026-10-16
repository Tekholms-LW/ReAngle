"""Keyframed camera paths (JSON) -> per-frame source->target transforms.

Same convention as camera.py: OpenCV camera (x right, y down, z forward); the ORIGINAL camera sits at
the origin looking down +z. A keyframe gives the new camera's centre ``pos`` and a ``look`` target, both in
original-camera coordinates (scene units; the pivot sits at z = pivot_depth). Keyframe 0 must be the
original camera: pos [0,0,0], look [0,0,pivot_depth].

JSON format (version 1)::

    {"version": 1, "pivot_depth": 3.0, "ease": "smooth" | "linear", "freeze_time": false,
     "keyframes": [{"t": 0.0, "pos": [0,0,0], "look": [0,0,3]}, {"t": 1.0, "pos": [...], "look": [...]}]}

t is normalised time (0 = first frame, 1 = last frame). Positions and look targets are interpolated with
Catmull-Rom splines over t. ``freeze_time`` asks the reprojection to use source frame 0 for every frame
("matrix freeze"). Paths are clamped to MAX_ANGLE_DEG of view rotation and MAX_TRAVEL x pivot_depth of travel.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .camera import MAX_ANGLE_DEG, camera_path

MAX_TRAVEL = 0.5          # x pivot_depth
DOWN = np.array([0.0, 1.0, 0.0])


def look_at_transform(pos, look) -> np.ndarray:
    """4x4 source->target transform for a camera at ``pos`` looking at ``look`` (no roll)."""
    pos, look = np.asarray(pos, float), np.asarray(look, float)
    f = look - pos
    f /= np.linalg.norm(f)
    r = np.cross(DOWN, f)
    if np.linalg.norm(r) < 1e-8:                      # looking straight up/down: pick any right vector
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    Rc = np.stack([r, d, f], axis=1)                  # camera axes in source coords
    T = np.eye(4)
    T[:3, :3] = Rc.T
    T[:3, 3] = -Rc.T @ pos
    return T


def transform_to_keyframe(T: np.ndarray, pivot_depth: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of look_at_transform (roll is dropped): returns (pos, look)."""
    R, t = T[:3, :3], T[:3, 3]
    pos = -R.T @ t
    fwd = R.T @ np.array([0.0, 0.0, 1.0])
    # look along forward at the distance where it is closest to the pivot (keeps orbits exact)
    p = np.array([0.0, 0.0, pivot_depth])
    s = max(float(np.dot(p - pos, fwd)), 0.25 * pivot_depth)
    return pos, pos + s * fwd


def _catmull(ts: np.ndarray, pts: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Non-uniform Catmull-Rom (cardinal, tension 0) through (ts, pts), evaluated at u."""
    if len(ts) == 1:
        return np.repeat(pts[:1], len(u), axis=0)
    out = np.empty((len(u), pts.shape[1]))
    for k, x in enumerate(u):
        i = int(np.clip(np.searchsorted(ts, x, side="right") - 1, 0, len(ts) - 2))
        t0, t1 = ts[i], ts[i + 1]
        p0, p1 = pts[i], pts[i + 1]
        m0 = (p1 - pts[i - 1]) / (t1 - ts[i - 1]) if i > 0 else (p1 - p0) / (t1 - t0)
        m1 = (pts[i + 2] - p0) / (ts[i + 2] - t0) if i + 2 < len(ts) else (p1 - p0) / (t1 - t0)
        h = t1 - t0
        s = (x - t0) / h
        h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
        out[k] = h00 * p0 + h10 * h * m0 + h01 * p1 + h11 * h * m1
    return out


def parse_path(spec: str | dict) -> dict:
    d = json.loads(spec) if isinstance(spec, str) else dict(spec)
    kfs = sorted(d.get("keyframes") or [], key=lambda k: float(k["t"]))
    if not kfs:
        raise ValueError("camera path needs at least one keyframe")
    piv = float(d.get("pivot_depth", 3.0))
    if piv <= 0:
        raise ValueError("pivot_depth must be > 0")
    ts = np.array([float(k["t"]) for k in kfs])
    if ts[0] != 0.0 or np.any(np.diff(ts) <= 0) or ts[-1] > 1.0:
        raise ValueError("keyframe times must start at 0, increase strictly and stay <= 1")
    pos = np.array([k["pos"] for k in kfs], float)
    look = np.array([k["look"] for k in kfs], float)
    if not (np.allclose(pos[0], 0, atol=1e-6) and np.allclose(look[0], [0, 0, piv], atol=1e-6)):
        raise ValueError("keyframe 0 must be the original camera: pos [0,0,0], look [0,0,pivot_depth]")
    return {"ts": ts, "pos": pos, "look": look, "pivot_depth": piv, "ease": d.get("ease", "smooth"),
            "freeze_time": bool(d.get("freeze_time", False))}


def clamp_pose(pos: np.ndarray, look: np.ndarray, pivot: float):
    """Limit travel to MAX_TRAVEL*pivot and view rotation to MAX_ANGLE_DEG. Returns (pos, look, clamped)."""
    clamped = False
    lim = MAX_TRAVEL * pivot
    n = np.linalg.norm(pos)
    if n > lim:
        pos = pos * (lim / n)
        clamped = True
    f = look - pos
    dist = np.linalg.norm(f)
    f = f / dist
    f0 = np.array([0.0, 0.0, 1.0])
    ang = np.arccos(np.clip(f @ f0, -1, 1))
    amax = np.deg2rad(MAX_ANGLE_DEG)
    if ang > amax + 1e-9:
        axis = np.cross(f0, f)
        axis /= np.linalg.norm(axis)
        c, s = np.cos(amax), np.sin(amax)        # rotate f0 by amax about axis (Rodrigues)
        f = f0 * c + np.cross(axis, f0) * s + axis * (axis @ f0) * (1 - c)
        clamped = True
    return pos, pos + dist * f, clamped


def path_from_json(spec: str | dict, num_frames: int) -> dict:
    """Returns {"poses": (N,4,4), "freeze_time": bool, "clamped_frames": int}."""
    p = parse_path(spec)
    u = np.linspace(0.0, 1.0, num_frames)
    if p["ease"] == "smooth":
        u = u * u * (3 - 2 * u)
    u = np.minimum(u, p["ts"][-1])              # hold the last keyframe if it ends before t=1
    P = _catmull(p["ts"], p["pos"], u)
    Lk = _catmull(p["ts"], p["look"], u)
    poses, nclamp = np.empty((num_frames, 4, 4)), 0
    for i in range(num_frames):
        pos, look, c = clamp_pose(P[i], Lk[i], p["pivot_depth"])
        nclamp += c
        poses[i] = look_at_transform(pos, look)
    poses[0] = np.eye(4)
    return {"poses": poses, "freeze_time": p["freeze_time"], "clamped_frames": int(nclamp)}


def preset_to_json(preset: str, angle_deg: float, pivot_depth: float = 3.0, keyframes: int = 7) -> dict[str, Any]:
    """Seed editable keyframes from a preset (used by the editor and for equivalence tests)."""
    T = camera_path(preset, angle_deg, keyframes, pivot_depth, ease="linear")
    kfs = []
    for i, Ti in enumerate(T):
        pos, look = transform_to_keyframe(Ti, pivot_depth)
        kfs.append({"t": i / (keyframes - 1), "pos": [round(float(v), 6) for v in pos],
                    "look": [round(float(v), 6) for v in look]})
    kfs[0] = {"t": 0.0, "pos": [0.0, 0.0, 0.0], "look": [0.0, 0.0, float(pivot_depth)]}
    return {"version": 1, "pivot_depth": pivot_depth, "ease": "smooth", "freeze_time": False, "keyframes": kfs}
