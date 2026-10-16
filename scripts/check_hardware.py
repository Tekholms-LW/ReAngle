#!/usr/bin/env python3
"""Setup check: what can this machine's ComfyUI do with ReAngle?  python scripts/check_hardware.py [--url URL] [--json]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from reshoot.hardware import detect, summary_text  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=None, help="ComfyUI URL (default COMFY_LOCAL_URL or http://127.0.0.1:8001)")
ap.add_argument("--hosted", action="store_true", help="treat as hosted ComfyUI")
ap.add_argument("--json", action="store_true")
a = ap.parse_args()
rep = detect(a.url, hosted=a.hosted or None)
print(json.dumps(rep, indent=1, default=str) if a.json else summary_text(rep))
sys.exit(0 if rep["can_render"] else 1)
