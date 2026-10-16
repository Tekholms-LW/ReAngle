# Submission kit: Comfy "Open Source Higgsfield"

Deadline: **Oct 30, 2026, 9:00 am PT (11:00 am CT)**. One submission per person. Winners announced Nov 6.
Requirements re-read from https://comfy.org/open-source-higgsfield on Oct 16, 2026.

## Checklist
- [ ] **GitHub repo** `Tekholms-LW/ReAngle`, public: app code ✓, workflow files ✓ (`workflows/`), OSS license ✓ (Apache-2.0), instructions to run locally ✓ (README). *Not created or pushed yet.*
- [ ] **Demo video ≤ 2:00** (see `docs/DEMO_SCRIPT.md`). Must explain why it's innovative (Most Innovative) and name all tools/models used.
- [ ] **Link to try it** (optional, encouraged): none yet. Options: a hosted Comfy API deployment (`RESHOOT_BACKEND=comfy`) or a GPU box; otherwise state "local only" and rely on the video.
- [ ] **Social post** linking the demo video on X/Reddit/IG/LI/YT/TikTok with **#OpenSourceHiggsfield**.
- [ ] Submit via the official submission form (the page links it; fields weren't visible without opening the form; check it a few days early).
- [ ] Built on the Comfy Developer Platform (Comfy API / Router / SDK). ReAngle drives ComfyUI through its API (local native API, or hosted Comfy API with `RESHOOT_BACKEND=comfy`). **Confirm this counts**, or test the hosted path once before submitting.
- [ ] Lawful/SFW, no unlicensed IP or likenesses: demo clip is the author's own Grok Imagine generation, no real person. Big Buck Bunny test clips are **not** in the repo.
- [ ] Categories to mention: Best Overall (auto), **Most Innovative** (3D path in the clip's own depth + background-aware fill), **Most Viral** (9:16 export), **Open Weights Only** (no partner nodes/APIs in local mode), **LTX VFX bonus** (LTX-2.5 is the core model).
- [ ] Before publishing: `git log` clean, no `.env`, no tokens, no personal paths (scanned Oct 16), media < 10 MB.

## Publish steps (when you approve)
1. Create the empty public repo `Tekholms-LW/ReAngle` on GitHub (no README/license, both exist locally).
2. `cd ReAngle && git remote add origin https://github.com/Tekholms-LW/ReAngle.git && git push -u origin main`
3. Optional: publish the node pack (`docs/PUBLISH_NODES.md`).
4. Upload the demo video (YouTube unlisted/public or X), then post.

## X post drafts (≤ 280 chars, one hashtag)
1. (239 chars)
> Filmed once, re-shot from a new angle on a 16GB GPU. ReAngle: draw a camera path inside your clip's own 3D depth, and LTX-2.5 re-renders it. Open weights, open source, built on @ComfyUI. #OpenSourceHiggsfield github.com/Tekholms-LW/ReAngle

2. (218 chars)
> Change the camera angle of any video, locally. ReAngle warps your clip along a path you draw in 3D, fills what the new view reveals, and lets @LTXStudio's LTX-2.5 paint the rest. Free on @ComfyUI. #OpenSourceHiggsfield

3. (204 chars)
> Before / after: same take, new camera. Open-source re-shoot app on @ComfyUI + LTX-2.5, safe on a 16GB card, one-click vertical export. Code + workflow: github.com/Tekholms-LW/ReAngle #OpenSourceHiggsfield

Attach the demo video (or `docs/media/social_stacked.mp4` as a teaser). Check the handles are right before posting (@ComfyUI, @LTXStudio).
