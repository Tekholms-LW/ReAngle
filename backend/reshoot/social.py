"""Vertical 9:16 before/after export for social posts (CPU / ffmpeg only, never touches the GPU).

Layouts
  stacked: original on top, re-shoot below, each fitted into a 1080x960 panel over a blurred fill of itself
  wipe:    one 1080x1920 frame; an animated vertical divider sweeps between original (left) and re-shoot (right)
Any source aspect works: the subject is fitted (never cropped), the leftover area is a blurred, darkened cover.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

W, H = 1080, 1920
CREDIT_PRESET = "Made with ReAngle · ComfyUI + LTX"
FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial.ttf",
]


def find_font() -> str | None:
    env = os.environ.get("RESHOOT_FONT")
    for f in ([env] if env else []) + FONT_CANDIDATES:
        if f and Path(f).is_file():
            return f
    return None


def _esc_path(p: str) -> str:
    """Path inside an ffmpeg filter argument: forward slashes, escape ':' and quotes."""
    return p.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def probe(p: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
                          "-of", "json", str(p)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    return {"duration": float(j["format"]["duration"]),
            "audio": any(s.get("codec_type") == "audio" for s in j.get("streams", []))}


def _panel(inp: str, w: int, h: int, tag: str) -> str:
    """Fit `inp` into w x h (no crop) over a blurred, darkened cover of itself."""
    return (f"[{inp}]split[{tag}a][{tag}b];"
            f"[{tag}a]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=24:2,"
            f"eq=brightness=-0.12[{tag}bg];"
            f"[{tag}b]scale={w}:{h}:force_original_aspect_ratio=decrease,setsar=1[{tag}fg];"
            f"[{tag}bg][{tag}fg]overlay=(W-w)/2:(H-h)/2,setsar=1[{tag}]")


def _text(tf: Path, font: str | None, x: str, y: str, size: int) -> str:
    f = f"fontfile='{_esc_path(font)}':" if font else ""
    return (f"drawtext={f}textfile='{_esc_path(str(tf))}':x={x}:y={y}:fontsize={size}:fontcolor=white:"
            f"box=1:boxcolor=black@0.45:boxborderw=14")


def build_filter(layout: str, labels: tuple[str, str], credit: str, tmp: Path, font: str | None) -> str:
    texts = {}
    for k, v in (("l0", labels[0]), ("l1", labels[1]), ("cr", credit)):
        if v:
            (tmp / f"{k}.txt").write_text(v, encoding="utf-8")
            texts[k] = tmp / f"{k}.txt"
    if layout == "stacked":
        g = _panel("0:v", W, H // 2, "top") + ";" + _panel("1:v", W, H // 2, "bot") + ";[top][bot]vstack=2[st]"
        last = "st"
        draws = []
        if "l0" in texts:
            draws.append(_text(texts["l0"], font, "40", "40", 56))
        if "l1" in texts:
            draws.append(_text(texts["l1"], font, "40", str(H // 2 + 40), 56))
    elif layout == "wipe":
        g = (_panel("0:v", W, H, "pa") + ";" + _panel("1:v", W, H, "pb") + ";"
             # divider sweeps 15% <-> 85% with a smooth sine; 6 px white line on the edge
             "[pa]format=gbrp[pa2];[pb]format=gbrp[pb2];[pa2][pb2]blend=all_expr='if(lt(abs(X-W*(0.5+0.35*sin(T*1.6))),3),255,"
             "if(lt(X,W*(0.5+0.35*sin(T*1.6))),A,B))'[st]")
        last = "st"
        draws = []
        if "l0" in texts:
            draws.append(_text(texts["l0"], font, "40", "60", 56))
        if "l1" in texts:
            draws.append(_text(texts["l1"], font, "w-tw-40", "60", 56))
    else:
        raise ValueError(f"unknown layout {layout!r}")
    if "cr" in texts:
        draws.append(_text(texts["cr"], font, "(w-tw)/2", "h-th-60", 36))
    if draws:
        g += f";[{last}]" + ",".join(draws) + "[out]"
        last = "out"
    return g + f";[{last}]format=yuv420p[v]"


def export_social(before: Path, after: Path, out: Path, layout: str = "stacked",
                  labels: tuple[str, str] = ("Before", "After"), credit: str = "", fps: int = 24) -> Path:
    """before = original clip, after = re-shoot (its length wins; the original is trimmed to match)."""
    a = probe(after)
    b = probe(before)
    dur = a["duration"]
    font = find_font()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fg = build_filter(layout, labels, credit.strip(), tmp, font)
        fg = fg.replace("[0:v]", f"[0:v]fps={fps},trim=duration={dur:.3f},setpts=PTS-STARTPTS,", 1)
        fg = fg.replace("[1:v]", f"[1:v]fps={fps},setpts=PTS-STARTPTS,", 1)
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(before), "-i", str(after),
               "-filter_complex", fg, "-map", "[v]"]
        if b["audio"]:
            cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "160k", "-af", f"atrim=duration={dur:.3f}"]
        elif a["audio"]:
            cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "160k"]
        cmd += ["-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                "-r", str(fps), "-movflags", "+faststart", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[-400:]}")
    return out
