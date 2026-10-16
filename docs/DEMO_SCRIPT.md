# ReAngle demo video (≤ 2:00)

Goal: show it working, say why it's innovative (required for *Most Innovative*), show it's open-weights only, LTX-core (LTX bonus), and shareable (*Most Viral*). Judges value the work over editing; record at 1080p, screen capture + voiceover is enough.

## Clips to record beforehand
| # | What | How |
|---|---|---|
| A | Hero before/after | `docs/media/before_after_orbit12.mp4` (or re-render) |
| B | Upload + aspect auto-pick | screen capture: drop `woman_fog_4s.mp4`, resolution shows "448x448 square (matches clip)" |
| C | 3D path editor | screen capture: Draw your own path → drag the camera, add a keyframe, scrub the warped preview, safety colours |
| D | Render progress | screen capture of chunk progress with "peak GPU … MB" (speed up 8–16×) |
| E | Fill on/off | 3-up `square_orbit_woman_compare.mp4` (original / before fill / with fill), freeze on the ghost frame |
| F | Hardware panel | screen capture: "Your hardware" tier, can/can't lists, model ✓ list |
| G | ComfyUI graph | ComfyUI canvas with the ReAngle nodes + LTX IC-LoRA guide (load `workflows/ltx25_reshoot_api.json`) |
| H | Social export | click Export for social → play the 9:16 stacked and wipe results (`docs/media/social_*.mp4`) |

## Shot list
| Time | Picture | Voiceover |
|---|---|---|
| 0:00–0:08 | A, full screen | "This shot was filmed once. ReAngle re-shot it from a new angle, on my own 16-gig GPU, with open weights." |
| 0:08–0:20 | B | "ReAngle is an open-source re-shoot app on ComfyUI. Drop a clip; it reads the aspect and your hardware and picks a safe resolution." |
| 0:20–0:45 | C | "Here's what's new: you direct the camera inside the clip's own 3D space. Video Depth Anything lifts the first frame into a depth mesh, I drag the camera, add keyframes, and see the warped view live. Presets are just starting points; the colours show what the model can handle." |
| 0:45–1:05 | G then D | "Under the hood: the whole clip is depth-warped along that path, and the warp guides LTX-2.5 through the Layout-to-Render IC-LoRA. Long clips render in overlapping chunks, each re-anchored to the source and colour-matched, so the subject doesn't drift." |
| 1:05–1:22 | E | "When the camera moves, it reveals things the original never saw. Passing black holes makes ghosts like this. ReAngle fills from the background side only and masks the guide there, so LTX paints the gap. Ghost gone." |
| 1:22–1:35 | F | "It's built to be safe on consumer cards: tier detection, a reserve-VRAM check, preflight, and a fast VRAM watchdog that stops ComfyUI before the display driver can crash." |
| 1:35–1:50 | H | "And one click makes a vertical before-and-after for social, stacked or as a wipe." |
| 1:50–2:00 | A again + repo URL | "Open weights only: LTX-2.5, Gemma, Video Depth Anything Small. Apache-2.0 code at github.com/Tekholms-LW/ReAngle. #OpenSourceHiggsfield" |

Tools/models to name on screen (rules require explaining them): ComfyUI, LTX-2.5 (GGUF Q4 via ComfyUI-GGUF), Layout-to-Render IC-LoRA, Gemma 4 text encoder, Video Depth Anything Small, ComfyUI-LTXVideo, VideoHelperSuite, three.js, FastAPI, ffmpeg.
