// 3D camera-path editor: the first frame as a depth-displaced mesh, draggable keyframes, live camera preview.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { CameraPath, Keyframe, MAX_ANGLE_DEG, MAX_TRAVEL, V3, clampPose, originalCamera, poseAt, validate } from "./path";

export type DepthPreview = { width: number; height: number; grid_w: number; grid_h: number; rgb: string; depth: string;
  hfov_deg: number; near: number; far: number; pivot_depth: number; source: string };

const cv = (p: V3) => new THREE.Vector3(p[0], -p[1], -p[2]);           // OpenCV -> three.js
const fromThree = (v: THREE.Vector3): V3 => [v.x, -v.y, -v.z];
const b64 = (s: string) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

export class PathEditor {
  path: CameraPath = { version: 1, pivot_depth: 3, ease: "smooth", freeze_time: false, keyframes: [originalCamera(3)] };
  playhead = 0;
  selected = 0;
  handle: "pos" | "look" = "pos";
  onChange: () => void = () => {};
  /** Called when the user tries to frame a shot while the playhead sits on the locked start. */
  onPlayheadJump: (u: number) => void = () => {};
  lastClamped = false;
  private fly?: { x: number; y: number; mode: "orbit" | "slide"; pos: V3; look: V3 };
  private scene = new THREE.Scene();
  private edCam: THREE.PerspectiveCamera;
  private pvCam: THREE.PerspectiveCamera;
  private edR: THREE.WebGLRenderer;
  private pvR: THREE.WebGLRenderer;
  private orbit: OrbitControls;
  private gizmo: TransformControls;
  private markers = new THREE.Group();
  private pathLine?: THREE.Line;
  private proxy = new THREE.Object3D();
  private dragging = false;
  private axes = new THREE.AxesHelper(0.3);

  constructor(private edEl: HTMLCanvasElement, private pvEl: HTMLCanvasElement) {
    this.scene.background = new THREE.Color(0x0b0d10);
    this.edR = new THREE.WebGLRenderer({ canvas: edEl, antialias: true });
    this.pvR = new THREE.WebGLRenderer({ canvas: pvEl, antialias: true });
    this.edCam = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
    this.edCam.position.set(4, 2.5, 2.5);
    this.pvCam = new THREE.PerspectiveCamera(40, 640 / 352, 0.05, 100);
    this.orbit = new OrbitControls(this.edCam, edEl);
    this.orbit.target.set(0, 0, -3);
    this.gizmo = new TransformControls(this.edCam, edEl);
    this.gizmo.setSize(0.7);
    this.gizmo.addEventListener("dragging-changed", (e) => { this.orbit.enabled = !e.value; this.dragging = !!e.value; if (!e.value) this.rebuild(); });
    this.gizmo.addEventListener("objectChange", () => this.applyProxy());
    this.scene.add(this.gizmo.getHelper(), this.markers, this.proxy);
    this.axes.name = "axes";
    this.scene.add(this.axes);
    new ResizeObserver(() => this.resize()).observe(edEl);
    new ResizeObserver(() => this.resize()).observe(pvEl);
    this.bindFly();
    this.resize();
    const loop = () => { this.render(); requestAnimationFrame(loop); };
    loop();
  }

  loadDepth(d: DepthPreview) {
    this.scene.getObjectByName("cloud")?.removeFromParent();
    const rgb = b64(d.rgb), gw = d.grid_w, gh = d.grid_h;
    let dep = Float32Array.from(b64(d.depth));
    for (let pass = 0; pass < 2; pass++) {                              // light 3x3 smoothing: fewer spikes in the preview mesh
      const o = new Float32Array(dep.length);
      for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
        let a = 0, n = 0;
        for (let y = Math.max(0, j - 1); y <= Math.min(gh - 1, j + 1); y++) for (let x = Math.max(0, i - 1); x <= Math.min(gw - 1, i + 1); x++) { a += dep[y * gw + x]; n++; }
        o[j * gw + i] = a / n;
      }
      dep = o;
    }
    const f = (0.5 * d.width) / Math.tan(((d.hfov_deg / 2) * Math.PI) / 180);
    const cx = (d.width - 1) / 2, cy = (d.height - 1) / 2;
    const pos = new Float32Array(gw * gh * 3), col = new Float32Array(gw * gh * 3), z = new Float32Array(gw * gh);
    for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
      const k = j * gw + i, r = dep[k] / 255;
      const inv = 1 / d.far + r * (1 / d.near - 1 / d.far);           // same mapping as depth_from_relative
      const zz = 1 / inv, u = ((i + 0.5) * d.width) / gw, v = ((j + 0.5) * d.height) / gh;
      const p = cv([((u - cx) / f) * zz, ((v - cy) / f) * zz, zz]);
      pos.set([p.x, p.y, p.z], k * 3); z[k] = zz;
      col.set([rgb[k * 3] / 255, rgb[k * 3 + 1] / 255, rgb[k * 3 + 2] / 255], k * 3);
    }
    const idx: number[] = [];
    const ok = (a: number, b: number) => Math.max(z[a], z[b]) / Math.min(z[a], z[b]) < 1.3; // tear only at real depth edges
    for (let j = 0; j < gh - 1; j++) for (let i = 0; i < gw - 1; i++) {
      const a = j * gw + i, b = a + 1, c = a + gw, e = c + 1;
      if (ok(a, b) && ok(a, c) && ok(b, c)) idx.push(a, c, b);
      if (ok(b, c) && ok(b, e) && ok(c, e)) idx.push(b, c, e);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.setIndex(idx);
    const mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
    mesh.name = "cloud";
    this.scene.add(mesh);
    this.pvCam.fov = (2 * Math.atan(Math.tan(((d.hfov_deg / 2) * Math.PI) / 180) * (d.height / d.width)) * 180) / Math.PI;
    this.pvCam.aspect = d.width / d.height;
    this.pvCam.updateProjectionMatrix();
    const shot = document.getElementById("shot");
    if (shot) {   // follow the clip's aspect (square/portrait) instead of a fixed 16:9 box
      shot.style.aspectRatio = `${d.width} / ${d.height}`;
      shot.style.maxWidth = `calc(70vh * ${d.width / d.height})`;
      shot.style.marginInline = "auto";
    }
    this.resize();
    this.setPath({ ...this.path, pivot_depth: d.pivot_depth, keyframes: [originalCamera(d.pivot_depth)] });
  }

  setPath(p: CameraPath) { this.lastClamped = false; this.path = JSON.parse(JSON.stringify(p)); this.selected = Math.min(this.selected, p.keyframes.length - 1); this.rebuild(); }

  /** Playhead (linear time) -> keyframe time (after the path's easing). */
  ease(u: number) { return this.path.ease === "smooth" ? u * u * (3 - 2 * u) : u; }
  /** Keyframe time -> playhead position. */
  unease(t: number) { return this.path.ease === "smooth" ? 0.5 - Math.sin(Math.asin(1 - 2 * t) / 3) : t; }

  /** Index of the keyframe under the playhead, or -1. */
  keyAtPlayhead() { const t = this.ease(this.playhead); return this.path.keyframes.findIndex((k) => Math.abs(k.t - t) < 0.015); }

  /** Store (pos, look) as the shot at the playhead: updates the keyframe there or inserts one. Clamps silently. */
  setShot(pos: V3, look: V3) {
    const c = clampPose(pos, look, this.path.pivot_depth);
    this.lastClamped = c.clamped;
    let i = this.keyAtPlayhead();
    if (i === 0) return;
    if (i < 0) {
      const t = Math.round(this.ease(this.playhead) * 1000) / 1000;
      this.path.keyframes.push({ t, pos: c.pos, look: c.look });
      this.path.keyframes.sort((a, b) => a.t - b.t);
      i = this.path.keyframes.findIndex((k) => k.t === t);
    } else this.path.keyframes[i] = { ...this.path.keyframes[i], pos: c.pos, look: c.look };
    this.selected = i;
    this.rebuild();
  }

  deleteAtPlayhead() { const i = this.keyAtPlayhead(); if (i > 0) { this.selected = i; this.deleteSelected(); } }
  reset() { this.selected = 0; this.path.freeze_time = false; this.setPath({ ...this.path, keyframes: [originalCamera(this.path.pivot_depth)] }); }

  private bindFly() {
    const el = this.pvEl, V = (a: V3) => new THREE.Vector3(...a), A = (v: THREE.Vector3): V3 => [v.x, v.y, v.z];
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    const begin = () => {
      if (this.playhead < 0.02) {                    // start is locked: continue the gesture on the end / last shot
        const k = this.path.keyframes, u = k.length > 1 ? this.unease(k[k.length - 1].t) : 1;
        this.playhead = u; this.onPlayheadJump(u);
      }
      return poseAt(this.path, this.playhead);
    };
    el.addEventListener("pointerdown", (e) => {
      const p = begin();
      this.fly = { x: e.clientX, y: e.clientY, mode: e.button === 2 || e.shiftKey ? "slide" : "orbit", pos: p.pos, look: p.look };
      el.setPointerCapture(e.pointerId); el.classList.add("grabbing");
    });
    el.addEventListener("pointermove", (e) => {
      const s = this.fly; if (!s) return;
      const w = el.clientWidth || 1, dx = (e.clientX - s.x) / w, dy = (e.clientY - s.y) / w;
      const pos = V(s.pos), look = V(s.look), f = look.clone().sub(pos).normalize();
      const right = new THREE.Vector3(0, 1, 0).cross(f).normalize(), down = f.clone().cross(right);  // OpenCV axes
      if (s.mode === "orbit") {                      // swing the camera around what it's looking at
        const v = pos.clone().sub(look);
        v.applyAxisAngle(new THREE.Vector3(0, 1, 0), dx * Math.PI * 0.6);
        v.applyAxisAngle(right, dy * Math.PI * 0.6);
        this.setShot(A(look.clone().add(v)), A(look));
      } else {                                       // slide / crane: move camera and target together
        const d = right.multiplyScalar(-dx).add(down.multiplyScalar(-dy)).multiplyScalar(this.path.pivot_depth * 1.2);
        this.setShot(A(pos.add(d)), A(look.add(d)));
      }
    });
    const end = (e: PointerEvent) => { if (this.fly) { this.fly = undefined; el.releasePointerCapture(e.pointerId); el.classList.remove("grabbing"); this.onChange(); } };
    el.addEventListener("pointerup", end); el.addEventListener("pointercancel", end);
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      const p = begin(), pos = V(p.pos), look = V(p.look), f = look.clone().sub(pos).normalize();
      this.setShot(A(pos.add(f.multiplyScalar(-Math.sign(e.deltaY) * 0.04 * this.path.pivot_depth))), p.look);
    }, { passive: false });
  }

  addKeyframeAtPlayhead() {
    const t = Math.round(this.ease(this.playhead) * 100) / 100;
    if (t <= 0 || this.path.keyframes.some((k) => Math.abs(k.t - t) < 0.02)) return false;
    const p = poseAt(this.path, this.playhead);
    this.path.keyframes.push({ t, pos: p.pos, look: p.look });
    this.path.keyframes.sort((a, b) => a.t - b.t);
    this.selected = this.path.keyframes.findIndex((k) => k.t === t);
    this.rebuild();
    return true;
  }

  deleteSelected() { if (this.selected > 0) { this.path.keyframes.splice(this.selected, 1); this.selected--; this.rebuild(); } }

  select(i: number) { this.selected = i; this.rebuild(); }

  stats() {
    let maxA = 0, maxT = 0, clamped = 0;
    for (let i = 0; i <= 60; i++) { const p = poseAt(this.path, i / 60); maxA = Math.max(maxA, p.angleDeg); maxT = Math.max(maxT, p.travel); clamped += +p.clamped; }
    return { maxAngle: maxA, maxTravel: maxT, clamped, error: validate(this.path) };
  }

  private kf(): Keyframe { return this.path.keyframes[this.selected]; }

  private applyProxy() {
    if (this.selected === 0) return;
    const k = this.kf();
    if (this.handle === "pos") k.pos = fromThree(this.proxy.position); else k.look = fromThree(this.proxy.position);
    this.rebuildMarkers();
    this.onChange();
  }

  rebuild() {
    this.rebuildMarkers();
    this.gizmo.detach();
    if (this.selected > 0) {
      const k = this.kf();
      this.proxy.position.copy(cv(this.handle === "pos" ? k.pos : k.look));
      this.gizmo.attach(this.proxy);
    }
    this.onChange();
  }

  private rebuildMarkers() {
    this.markers.clear();
    this.path.keyframes.forEach((k, i) => {
      const sel = i === this.selected;
      const color = i === 0 ? 0x9aa0aa : sel ? 0x6ee7b7 : 0xf5c26b;
      const cam = new THREE.Mesh(new THREE.ConeGeometry(0.08, 0.16, 4, 1, true),
        new THREE.MeshBasicMaterial({ color, wireframe: true }));
      cam.geometry.rotateX(-Math.PI / 2);                         // cone apex at camera, opening toward -z
      cam.geometry.translate(0, 0, 0.08);                         // apex at the camera centre
      cam.position.copy(cv(k.pos)); cam.lookAt(cv(k.look));
      const tgt = new THREE.Mesh(new THREE.SphereGeometry(0.035), new THREE.MeshBasicMaterial({ color }));
      tgt.position.copy(cv(k.look));
      const ray = new THREE.Line(new THREE.BufferGeometry().setFromPoints([cv(k.pos), cv(k.look)]),
        new THREE.LineDashedMaterial({ color, dashSize: 0.05, gapSize: 0.04 }));
      ray.computeLineDistances();
      this.markers.add(cam, tgt, ray);
    });
    const pts: THREE.Vector3[] = [], cols: number[] = [];
    for (let i = 0; i <= 80; i++) {
      const p = poseAt(this.path, i / 80);
      pts.push(cv(p.pos));
      const warn = p.angleDeg > MAX_ANGLE_DEG * 0.8 || p.travel > MAX_TRAVEL * 0.8;
      cols.push(...(p.clamped ? [1, 0.25, 0.25] : warn ? [1, 0.75, 0.2] : [0.43, 0.9, 0.72]));
    }
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    g.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
    this.pathLine = new THREE.Line(g, new THREE.LineBasicMaterial({ vertexColors: true }));
    this.markers.add(this.pathLine);
    const safe = new THREE.Mesh(new THREE.SphereGeometry(MAX_TRAVEL * this.path.pivot_depth, 24, 12),
      new THREE.MeshBasicMaterial({ color: 0x6ee7b7, wireframe: true, transparent: true, opacity: 0.08 }));
    this.markers.add(safe);
    const p = poseAt(this.path, this.playhead);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.04), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    head.position.copy(cv(p.pos));
    this.markers.add(head);
  }

  setPlayhead(u: number) { this.playhead = u; if (!this.dragging) this.rebuildMarkers(); }

  private resize() {
    const w = this.edEl.clientWidth, h = this.edEl.clientHeight;
    if (w && h) { this.edR.setSize(w, h, false); this.edCam.aspect = w / h; this.edCam.updateProjectionMatrix(); }
    if (this.pvEl.clientWidth) this.pvR.setSize(this.pvEl.clientWidth, this.pvEl.clientHeight, false);
  }

  private render() {
    const p = poseAt(this.path, this.playhead);
    this.pvCam.position.copy(cv(p.pos));
    this.pvCam.up.set(0, 1, 0);
    this.pvCam.lookAt(cv(p.look));
    this.markers.visible = false; this.axes.visible = false; this.gizmo.getHelper().visible = false;
    this.pvR.render(this.scene, this.pvCam);
    this.markers.visible = true; this.axes.visible = true; this.gizmo.getHelper().visible = this.selected > 0;
    if (this.edEl.clientWidth) this.edR.render(this.scene, this.edCam);
  }

  setHandle(h: "pos" | "look") { this.handle = h; this.rebuild(); }
}
