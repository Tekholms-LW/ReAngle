"""Camera-move presets exposed to the UI. Angles are capped at MAX_ANGLE_DEG."""
from __future__ import annotations

MAX_ANGLE_DEG = 30.0

PRESETS: dict[str, dict] = {
    "orbit_left":  {"label": "Orbit left",  "desc": "Arc around the subject to the left"},
    "orbit_right": {"label": "Orbit right", "desc": "Arc around the subject to the right"},
    "low":         {"label": "Low angle",   "desc": "Camera drops and tilts up"},
    "high":        {"label": "High angle",  "desc": "Camera rises and tilts down"},
    "push":        {"label": "Push in",     "desc": "Dolly toward the subject"},
    "pull":        {"label": "Pull out",    "desc": "Dolly away from the subject"},
    "truck":       {"label": "Truck",       "desc": "Slide sideways, parallel to the subject"},
    "crane":       {"label": "Crane",       "desc": "Rise vertically over the shot"},
}


def clamp_strength(deg: float) -> float:
    """Signed angle/strength in degrees, clamped to +-MAX_ANGLE_DEG."""
    return max(-MAX_ANGLE_DEG, min(MAX_ANGLE_DEG, float(deg)))
