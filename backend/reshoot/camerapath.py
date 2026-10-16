"""Bridge to the node pack's camera-path code so the backend validates exactly what ComfyUI will run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_NODES = Path(__file__).resolve().parents[2] / "comfyui-reshoot-nodes"
if str(_NODES) not in sys.path:
    sys.path.insert(0, str(_NODES))

from reshoot_nodes.camera_json import parse_path, path_from_json, preset_to_json  # noqa: E402,F401


def normalize_path(spec: dict) -> str:
    """Validate a custom path from the editor and return compact JSON (raises ValueError)."""
    if len(json.dumps(spec)) > 20000 or len(spec.get("keyframes") or []) > 32:
        raise ValueError("camera path too large (max 32 keyframes)")
    parse_path(spec)
    return json.dumps(spec, separators=(",", ":"))


def approx_yaw_deg(path_json: str) -> float:
    """Signed yaw of the final keyframe (used only by the mock renderer)."""
    import numpy as np
    p = parse_path(path_json)
    f = p["look"][-1] - p["pos"][-1]
    return float(np.degrees(np.arctan2(f[0], f[2])))
