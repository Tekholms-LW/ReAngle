# comfyui-reangle

ComfyUI nodes from [ReAngle](https://github.com/Tekholms-LW/ReAngle): re-shoot a clip from a new camera angle.

| Node | What it does |
|---|---|
| **Camera Path From Preset (ReAngle)** | preset (orbit, low/high, push/pull, truck, crane) + signed angle (±30°) → per-frame 4×4 camera transforms |
| **Camera Path From JSON (ReAngle)** | a keyframed path drawn in the ReAngle 3D editor (Catmull-Rom + easing, optional freeze-time) → per-frame transforms |
| **Depth Reproject (ReAngle)** | frames + relative depth (Video Depth Anything) + path → warped frames, hole mask, warped depth (z-buffered splat, crack fill) |

`reshoot_nodes/reproject.py` also has `prefill_holes`, the background-aware fill that ReAngle uses so revealed areas never reach LTX as black: far-side fill for disocclusions, background-only mirroring at frame borders, see-through speck removal, and a feathered regen mask for `LTXAddVideoICLoRAGuideAdvanced.attention_mask`.

Pure numpy, CPU-testable: `python -m pytest -q tests`. Apache-2.0.
