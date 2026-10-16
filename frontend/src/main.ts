import { PathEditor, DepthPreview } from "./editor";
import type { CameraPath } from "./path";

type Preset = { id: string; label: string; desc: string };
type Angle = { preset: string; angle_deg: number; status: string; progress: number; output: string | null; error: string | null;
  detail?: string | null; stats?: { peak_vram_mb?: number; chunks?: unknown } | null };
type Job = { id: string; status: string; progress: number; angles: Angle[]; upload_id: string; opts?: { frames: number; chunked?: boolean } };
type Model = { role: string; file: string; folder: string; size_gb: number; present: boolean; required: boolean; url: string };
type Tier = { id: string; label: string; why: string; supported: boolean; resolutions: string[]; chunk_frames: number; overlap: number;
  max_seconds: number; upscale: "yes" | "short" | "no"; upscale_max_frames: number; reserve_vram: number };
type Hardware = { gpu: string | null; vram_total_mb: number; vram_free_mb: number; torch?: string; cuda?: string; ram_total_gb?: number;
  reserve_vram: number | null; tier: Tier; caps: { can: string[]; cannot: string[]; resolutions_by_aspect?: Record<string, string[]> }; models: Model[]; warnings: string[];
  can_render: boolean; unet: string | null; mock?: boolean; hosted?: boolean; nvidia_smi?: { driver: string } | null };

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const api = async <T>(path: string, init?: RequestInit): Promise<T> => {
  const r = await fetch(path, init);
  if (!r.ok) {
    const t = await r.text();
    let msg = t; try { msg = JSON.parse(t).detail ?? t; } catch { /* plain text */ }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
};

const selected = new Set<string>(["orbit_left"]);
let uploadId: string | null = null;
let srcAspect = "landscape", aspectChanged = false;
let originalUrl = "";
let mode: "presets" | "draw" = "presets";
let editor: PathEditor | null = null;
let clipFrames = 41, fps = 24, srcSeconds = 1.7;
let hw: Hardware | null = null;
let currentJob: string | null = null;

async function init() {
  const health = await api<{ backend: string }>("/api/health");
  $("backend").textContent = `backend: ${health.backend}`;
  loadHardware();
  const { presets, max_angle_deg } = await api<{ presets: Preset[]; max_angle_deg: number }>("/api/presets");
  $<HTMLInputElement>("angle").max = String(max_angle_deg);
  const box = $("presets");
  for (const p of presets) {
    const b = document.createElement("button");
    b.className = "chip" + (selected.has(p.id) ? " on" : "");
    b.textContent = p.label; b.title = p.desc; b.type = "button";
    b.onclick = () => { selected.has(p.id) ? selected.delete(p.id) : selected.add(p.id); b.classList.toggle("on"); refresh(); };
    box.append(b);
  }
}

function refresh() {
  if (hw && !hw.can_render) { $<HTMLButtonElement>("go").disabled = true; $<HTMLButtonElement>("go").title = "This machine can't render yet; see Your hardware"; return; }
  const drawOk = mode === "draw" && editor && !editor.stats().error && editor.path.keyframes.length > 1;
  $<HTMLButtonElement>("go").disabled = !uploadId || (mode === "presets" ? selected.size === 0 : !drawOk);
}

function setMode(m: "presets" | "draw") {
  mode = m;
  $("mode-presets").classList.toggle("on", m === "presets"); $("mode-draw").classList.toggle("on", m === "draw");
  $("presets-pane").hidden = m !== "presets"; $("draw-pane").hidden = m !== "draw";
  if (m === "draw") { ensureEditor(); if (!localStorage.getItem("reshoot.tour")) showTour(true); }
  refresh();
}
$("mode-presets").onclick = () => setMode("presets");
$("mode-draw").onclick = () => setMode("draw");

const dur = () => (clipFrames - 1) / fps;
const fmt = (u: number) => `${(u * dur()).toFixed(1)} s`;
function showTour(v: boolean) { $("tour").hidden = !v; if (!v) localStorage.setItem("reshoot.tour", "1"); }
$("help").onclick = () => showTour(true);
$("tour-ok").onclick = () => showTour(false);

function ensureEditor() {
  if (!editor) {
    editor = new PathEditor($<HTMLCanvasElement>("ed-canvas"), $<HTMLCanvasElement>("pv-canvas"));
    editor.onChange = () => { renderTimeline(); renderSafety(); refresh(); };
    editor.onPlayheadJump = (u) => jumpTo(u, true);
  }
  if (uploadId) loadDepth();
}

let depthFor = "";
async function loadDepth() {
  if (!editor || !uploadId || depthFor === uploadId) return;
  depthFor = uploadId;
  $("shot-empty").hidden = false; $("shot-empty").textContent = "Building a 3D view of your first frame…";
  try {
    const d = await api<DepthPreview>(`/api/uploads/${uploadId}/depth`);
    editor.loadDepth(d);
    $("shot-empty").hidden = true;
    $("tl-end").textContent = fmt(1);
    tl.value = "1000"; scrub();
  } catch (e) { depthFor = ""; $("shot-empty").textContent = `Couldn't build the 3D view: ${(e as Error).message}`; }
}

/** Visibly move the playhead (animated + highlight). editorSet=true when the editor already moved its own playhead. */
let jumpAnim = 0;
function jumpTo(u: number, editorSet = false) {
  cancelAnimationFrame(jumpAnim);
  const from = Number(tl.value) / 1000, t0 = performance.now();
  $("tl-wrap").classList.remove("pulse"); void $("tl-wrap").offsetWidth; $("tl-wrap").classList.add("pulse");
  flash(`Framing your end shot at ${fmt(u)} (the start stays your original camera)`);
  const step = (now: number) => {
    const a = Math.min(1, (now - t0) / 350), e = a * (2 - a), v = from + (u - from) * e;
    tl.value = String(Math.round(v * 1000));
    $("tl-out").textContent = fmt(v);
    if (!editorSet) editor?.setPlayhead(v);
    if (a < 1) jumpAnim = requestAnimationFrame(step); else { tl.value = String(Math.round(u * 1000)); scrub(); }
  };
  jumpAnim = requestAnimationFrame(step);
}
const endTarget = () => { const k = editor?.path.keyframes ?? []; return k.length > 1 ? editor!.unease(k[k.length - 1].t) : 1; };
$("set-end").onclick = () => jumpTo(endTarget());

let hintTimer = 0;
function flash(msg: string) {
  const h = $("draw-hint"); h.textContent = msg; h.classList.add("flash");
  clearTimeout(hintTimer); hintTimer = window.setTimeout(() => { h.classList.remove("flash"); updateHint(); }, 3500);
}
function updateHint() {
  if (!editor || $("draw-hint").classList.contains("flash")) return;
  const n = editor.path.keyframes.length;
  $("draw-hint").textContent = editor.playhead < 0.02
    ? (n > 1 ? "Press Play to preview · move the timeline to change a shot" : "Start is your original camera. Drag the picture or press “Set end shot →” to frame the end")
    : n === 1 ? "Drag to frame your end shot"
    : editor.keyAtPlayhead() > 0 ? `Shot at ${fmt(editor.playhead)} saved · drag to adjust it, or press Play`
    : `Drag to set the shot at ${fmt(editor.playhead)}`;
}

async function seed(kind: string) {
  ensureEditor(); if (!editor || !uploadId) return;
  const preset = kind === "matrix" ? "orbit_left" : kind;
  const p = await api<CameraPath>(`/api/presets/${preset}/path?angle_deg=${Math.max(Number(angle.value), 10)}`);
  const k = p.keyframes, mid = k[Math.floor(k.length / 2)];
  p.keyframes = [k[0], mid, k[k.length - 1]];          // start, middle, end: easy to tweak
  p.freeze_time = kind === "matrix";
  $<HTMLInputElement>("freeze").checked = p.freeze_time;
  editor.setPath(p);
  play(true);
}
document.querySelectorAll<HTMLButtonElement>("#moves button").forEach((b) => (b.onclick = () => seed(b.dataset.move!)));

function renderTimeline() {
  if (!editor) return;
  const box = $("tl-marks"); box.replaceChildren();
  editor.path.keyframes.forEach((k, i) => {
    const u = editor!.unease(k.t), d = document.createElement("button");
    d.type = "button"; d.className = "mark" + (i === 0 ? " start" : "") + (i === editor!.keyAtPlayhead() ? " on" : "");
    d.style.left = `${u * 100}%`;
    d.title = i === 0 ? "Start: your original camera (locked). The re-shoot has to begin from the real camera so the first frame matches your footage; the move starts from here." : `Shot at ${fmt(u)}`;
    if (i === 0) d.textContent = "🔒";
    d.onclick = () => { tl.value = String(Math.round(u * 1000)); scrub(); };
    box.append(d);
  });
  const i = editor.keyAtPlayhead();
  $("kf-del").hidden = i <= 0;
  $("start-lock").hidden = editor.playhead >= 0.02 || !!playing;
  $("shot-label").textContent = editor.playhead < 0.02 ? "Start · original camera (locked)" : i > 0 ? `Shot at ${fmt(editor.playhead)} ✓` : `Preview at ${fmt(editor.playhead)}`;
  updateHint();
}

function renderSafety() {
  if (!editor) return;
  const s = editor.stats();
  const lvl = Math.max(s.maxAngle / 30, s.maxTravel / 0.5);
  const far = s.clamped > 0 || editor.lastClamped || lvl > 0.97, stretch = !far && lvl > 0.7;
  $("meter").className = "meter " + (far ? "bad" : stretch ? "warn" : "ok");
  $<HTMLElement>("meter-bar").style.width = `${Math.min(100, lvl * 100)}%`;
  $("meter-text").textContent = far ? "Too far" : stretch ? "Stretching" : "Safe";
  $("meter").title = far ? "We limit the move to 30° of turn and a short slide so the re-shoot stays clean. Your shot was held at that limit."
    : stretch ? "Bigger moves reveal more hidden areas for the model to invent." : "A comfortable move for a clean re-shoot.";
}

const tl = $<HTMLInputElement>("tl");
function scrub() {
  const u = Number(tl.value) / 1000;
  editor?.setPlayhead(u);
  $("tl-out").textContent = fmt(u);
  renderTimeline();
}
tl.oninput = scrub;
let playing = 0;
function play(fromStart = false) {
  if (playing) { cancelAnimationFrame(playing); playing = 0; $("tl-play").textContent = "▶ Play"; return; }
  if (fromStart || Number(tl.value) >= 999) tl.value = "0";
  $("tl-play").textContent = "❚❚ Pause";
  const ms = dur() * 1000, t0 = performance.now() - (Number(tl.value) / 1000) * ms;
  const step = (now: number) => {
    const u = Math.min(1, (now - t0) / ms); tl.value = String(Math.round(u * 1000)); scrub();
    if (u >= 1) { playing = 0; $("tl-play").textContent = "▶ Play"; return; }
    playing = requestAnimationFrame(step);
  };
  playing = requestAnimationFrame(step);
}
$("tl-play").onclick = () => play();
$("reset").onclick = () => { editor?.reset(); $<HTMLInputElement>("freeze").checked = false; tl.value = "1000"; scrub(); };
$("kf-del").onclick = () => editor?.deleteAtPlayhead();
$("h-pos").onclick = () => { editor?.setHandle("pos"); $("h-pos").classList.add("on"); $("h-look").classList.remove("on"); };
$("h-look").onclick = () => { editor?.setHandle("look"); $("h-look").classList.add("on"); $("h-pos").classList.remove("on"); };
$<HTMLInputElement>("freeze").onchange = (e) => { if (editor) { editor.path.freeze_time = (e.target as HTMLInputElement).checked; editor.onChange(); } };
$("advanced").addEventListener("toggle", () => { const i = editor?.keyAtPlayhead() ?? -1; if (editor && i > 0) editor.select(i); });

const esc = (t: string) => t.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
async function loadHardware(refresh = false) {
  try { hw = await api<Hardware>(`/api/hardware${refresh ? "?refresh=true" : ""}`); } catch (e) { $("hw-sum").textContent = `unknown (${(e as Error).message})`; return; }
  const t = hw.tier, gb = (mb: number) => (mb / 1024).toFixed(1);
  $("hw-sum").innerHTML = `${esc(hw.gpu ?? "unknown GPU")}${hw.vram_total_mb ? ` · ${gb(hw.vram_total_mb)} GB` : ""} · <b class="${hw.can_render ? "ok" : "bad"}">${esc(t.label)}</b>` +
    (hw.can_render ? "" : ` · <b class="bad">not ready</b>`);
  const models = hw.models.map((m) => `<tr class="${m.present ? "ok" : m.required ? "bad" : ""}"><td>${m.present ? "✓" : m.required ? "✕ missing" : "optional"}</td>
    <td><code>${esc(m.file)}</code><br><span class="hint">${esc(m.role)} · ${m.size_gb} GB · models/${esc(m.folder)}/</span></td>
    <td>${m.present ? "" : `<a href="${esc(m.url)}" target="_blank" rel="noopener">download</a>`}</td></tr>`).join("");
  $("hw-body").innerHTML = `
    <div class="hw-grid">
      <div><span class="hint">GPU</span><b>${esc(hw.gpu ?? "-")}</b></div>
      <div><span class="hint">VRAM</span><b>${hw.vram_total_mb ? `${gb(hw.vram_total_mb)} GB (${gb(hw.vram_free_mb)} free)` : "-"}</b></div>
      <div><span class="hint">torch / CUDA</span><b>${esc(hw.torch ?? "-")}</b></div>
      <div><span class="hint">System RAM</span><b>${hw.ram_total_gb ? hw.ram_total_gb + " GB" : "-"}</b></div>
      <div><span class="hint">--reserve-vram</span><b>${hw.reserve_vram ?? "-"} (needs ${t.reserve_vram})</b></div>
      <div><span class="hint">Model</span><b>${esc(hw.unet ?? (hw.mock ? "mock" : "none installed"))}</b></div>
    </div>
    <p class="tier-why"><b>${esc(t.label)}:</b> ${esc(t.why)}</p>
    <div class="cancan"><div><h4>Can</h4><ul>${hw.caps.can.map((x) => `<li>✓ ${esc(x)}</li>`).join("")}</ul></div>
      <div><h4>Can't</h4><ul>${hw.caps.cannot.map((x) => `<li>✕ ${esc(x)}</li>`).join("") || "<li>-</li>"}</ul></div></div>
    ${hw.warnings.map((w) => `<p class="notice">⚠ ${esc(w)}</p>`).join("")}
    ${models ? `<table class="models">${models}</table>` : ""}
    <p class="hint">Setup check from a terminal: <code>python scripts/check_hardware.py</code> · <a href="#" id="hw-refresh">re-check</a></p>`;
  $("hw-refresh").onclick = (e) => { e.preventDefault(); loadHardware(true); };
  if (!hw.can_render) $<HTMLDetailsElement>("hw-details").open = true;
  applyTier();
}

const durIn = $<HTMLInputElement>("dur"), res = $<HTMLSelectElement>("res"), up = $<HTMLInputElement>("upscale");
function applyTier() {
  if (!hw) return;
  const t = hw.tier;
  const max = Math.max(1, Math.min(t.max_seconds || 1, srcSeconds));
  durIn.max = String(max);
  if (Number(durIn.value) > max) durIn.value = String(max);
  const prev = res.value;
  const by = hw.caps?.resolutions_by_aspect ?? { landscape: t.resolutions };
  const order = [srcAspect, ...["landscape", "square", "portrait"].filter((a) => a !== srcAspect)];
  const opts: HTMLOptionElement[] = [];
  for (const a of order) for (const r of by[a] ?? []) {
    const o = document.createElement("option"); o.value = r;
    o.textContent = `${r} ${a}` + (a === srcAspect ? " (matches clip)" : ""); opts.push(o);
  }
  res.replaceChildren(...opts);
  if (prev && opts.some((o) => o.value === prev) && !aspectChanged) res.value = prev;
  aspectChanged = false;
  res.disabled = opts.length < 2;
  updateQuality();
}
function updateQuality() {
  if (!hw) return;
  const t = hw.tier, secs = Number(durIn.value), frames = Math.round(secs * fps);
  $("dur-out").textContent = `${secs.toFixed(1)} s`;
  const upOk = t.upscale === "yes" ? frames <= t.upscale_max_frames : t.upscale === "short" ? frames <= t.upscale_max_frames : false;
  up.disabled = !upOk; if (!upOk) up.checked = false;
  $("up-wrap").title = t.upscale === "no" ? "Not available on this GPU" : upOk ? "Renders at 2x resolution (slower)" :
    `On this GPU, 2x upscale works for clips up to ${(t.upscale_max_frames / fps).toFixed(1)} s`;
  const chunks = frames > t.chunk_frames ? Math.ceil((frames - t.overlap) / (t.chunk_frames - t.overlap)) : 1;
  $("q-hint").textContent = !t.supported ? t.why : `${frames} frames` + (chunks > 1 ? ` in ${chunks} overlapping chunks` : "") +
    (up.checked ? ` · upscaled to ${res.value.split("x").map((v) => Number(v) * 2).join("x")}` : "") +
    ` · max ${t.max_seconds} s on ${t.label.split(" (")[0]}` + (t.upscale === "short" && !upOk ? ` · upscale needs ≤ ${(t.upscale_max_frames / fps).toFixed(1)} s` : "");
  clipFrames = Math.max(9, frames);
  if (editor) { $("tl-end").textContent = fmt(1); scrub(); }
}
durIn.oninput = updateQuality; up.onchange = updateQuality; res.onchange = updateQuality;

async function handleFile(f: File) {
  $("drop-text").textContent = `Uploading ${f.name}…`;
  const fd = new FormData(); fd.append("file", f);
  try {
    const r = await api<{ upload_id: string; url: string; notice?: string; aspect?: string; source?: { frames: number; fps: number } }>("/api/uploads", { method: "POST", body: fd });
    if ((r.aspect ?? "landscape") !== srcAspect) { srcAspect = r.aspect ?? "landscape"; aspectChanged = true; }
    const sf = r.source?.frames ?? 41, sfps = r.source?.fps ?? 24;
    srcSeconds = sf / sfps;
    durIn.value = String(Math.min(Number(durIn.max) || 5, Math.max(1, Math.round(Math.min(srcSeconds, 41 / fps) * 2) / 2)));
    applyTier();
    uploadId = r.upload_id; originalUrl = r.url;
    const v = $<HTMLVideoElement>("preview"); v.src = r.url; v.hidden = false;
    $("drop-text").textContent = `${f.name} uploaded. Drop another to replace.`;
    const n = $("notice"); n.textContent = r.notice ?? ""; n.hidden = !r.notice;
    if (mode === "draw") loadDepth();
  } catch (e) { $("drop-text").textContent = `Upload failed: ${(e as Error).message}`; }
  refresh();
}

const file = $<HTMLInputElement>("file"), drop = $("drop");
file.onchange = () => file.files?.[0] && handleFile(file.files[0]);
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer?.files[0]; if (f) handleFile(f); };

const angle = $<HTMLInputElement>("angle");
angle.oninput = () => ($("angle-out").textContent = `${angle.value}°`);

$("go").onclick = async () => {
  if (!uploadId) return;
  $<HTMLButtonElement>("go").disabled = true;
  $("progress").hidden = false;
  const angles = mode === "draw" && editor
    ? [{ preset: "custom", angle_deg: 0, path: editor.path }]
    : [...selected].map((preset) => ({ preset, angle_deg: Number(angle.value) }));
  try {
    let job = await api<Job>("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ upload_id: uploadId, angles, duration_s: Number(durIn.value), resolution: res.value || undefined,
        upscale: up.checked, fill: $<HTMLInputElement>("fill").checked }) });
    await follow(job);
  } catch (e) { $("progress-text").textContent = `Error: ${(e as Error).message}`; }
  refresh();
};

async function follow(job: Job) {
  currentJob = job.id;
  $("cancel").hidden = false; $("resume").hidden = true;
  while (job.status === "queued" || job.status === "running") {
    setProgress(job);
    await new Promise((r) => setTimeout(r, 700));
    job = await api<Job>(`/api/jobs/${job.id}`);
  }
  setProgress(job);
  $("cancel").hidden = true;
  const errs = job.angles.filter((a) => a.error).map((a) => `${a.preset}: ${a.error}`);
  if (errs.length) $("progress-text").textContent += ` · ${errs.join(" · ")}`;
  $("resume").hidden = !job.angles.some((a) => a.status === "error" || a.status === "cancelled");
  if (job.angles.some((a) => a.status === "done")) showResults(job);
}
$("cancel").onclick = async () => { if (currentJob) { await api(`/api/jobs/${currentJob}/cancel`, { method: "POST" }); $("progress-detail").textContent = "Cancelling… (stops after the current GPU step)"; } };
$("resume").onclick = async () => {
  if (!currentJob) return;
  try { const j = await api<Job>(`/api/jobs/${currentJob}/resume`, { method: "POST" }); await follow(j); }
  catch (e) { $("progress-text").textContent = `Error: ${(e as Error).message}`; }
};

function setProgress(j: Job) {
  $("bar-fill").style.width = `${Math.round(j.progress * 100)}%`;
  const done = j.angles.filter((a) => a.status === "done").length;
  $("progress-text").textContent = `${j.status} · ${done}/${j.angles.length} angles · ${Math.round(j.progress * 100)}%`;
  const cur = j.angles.find((a) => a.status === "running") ?? j.angles[j.angles.length - 1];
  const peak = cur?.stats?.peak_vram_mb;
  $("progress-detail").textContent = (cur?.detail ?? "") + (peak ? ` · peak GPU ${peak} MB` : "");
}

function showResults(j: Job) {
  $("step-results").hidden = false;
  const tabs = $("tabs"); tabs.replaceChildren();
  const a = $<HTMLVideoElement>("vid-a"), b = $<HTMLVideoElement>("vid-b");
  a.src = originalUrl;
  const items: { label: string; url: string }[] = [{ label: "Original", url: originalUrl }];
  for (const x of j.angles) {
    if (x.status === "done" && x.output) items.push({ label: x.preset === "custom" ? "Your path" : `${x.preset.replace("_", " ")} ${x.angle_deg}°`, url: `/api/outputs/${x.output}` });
    else items.push({ label: `${x.preset} (failed)`, url: "" });
  }
  const select = (i: number) => {
    [...tabs.children].forEach((c, k) => c.classList.toggle("on", k === i));
    b.src = items[i].url; const dl = $<HTMLAnchorElement>("download");
    dl.href = items[i].url + (i ? "?download=true" : ""); syncPlay();
  };
  items.forEach((it, i) => {
    const t = document.createElement("button"); t.className = "chip"; t.type = "button"; t.textContent = it.label;
    t.disabled = !it.url; t.onclick = () => select(i); tabs.append(t);
  });
  select(items.length > 1 ? 1 : 0);
  socialJob = j; socialAngle = 0;
}

let socialJob: Job | null = null, socialAngle = 0;
$<HTMLInputElement>("soc-preset").onchange = (e) => {
  $<HTMLInputElement>("soc-credit").value = (e.target as HTMLInputElement).checked ? "Made with ReAngle · ComfyUI + LTX" : "";
};
$("soc-go").onclick = async () => {
  if (!socialJob) return;
  const idx = socialJob.angles.findIndex((x) => x.status === "done" && x.output);
  socialAngle = idx < 0 ? 0 : idx;
  const btn = $<HTMLButtonElement>("soc-go"); btn.disabled = true; $("soc-msg").textContent = "Rendering 1080x1920 on the CPU…";
  try {
    const r = await api<{ url: string; download: string }>(`/api/jobs/${socialJob.id}/social`, { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ angle: socialAngle, layout: $<HTMLSelectElement>("soc-layout").value,
        labels: $<HTMLSelectElement>("soc-labels").value.split("|"), credit: $<HTMLInputElement>("soc-credit").value }) });
    const dl = $<HTMLAnchorElement>("soc-dl"); dl.href = r.download; dl.hidden = false; $("soc-msg").textContent = "Ready.";
  } catch (e) { $("soc-msg").textContent = `Export failed: ${(e as Error).message}`; }
  btn.disabled = false;
};

function syncPlay() {
  const a = $<HTMLVideoElement>("vid-a"), b = $<HTMLVideoElement>("vid-b");
  b.currentTime = a.currentTime; a.play().catch(() => {}); b.play().catch(() => {});
}
$("play").onclick = () => {
  const a = $<HTMLVideoElement>("vid-a"), b = $<HTMLVideoElement>("vid-b");
  if (a.paused) syncPlay(); else { a.pause(); b.pause(); }
};
const wipe = $<HTMLInputElement>("wipe");
wipe.oninput = () => {
  $("clip").style.clipPath = `inset(0 0 0 ${wipe.value}%)`;
  $("compare").style.setProperty("--x", `${wipe.value}%`);
};

init().catch((e) => ($("backend").textContent = `API offline: ${e.message}`));
