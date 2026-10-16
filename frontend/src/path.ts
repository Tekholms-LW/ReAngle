// Camera-path math — a TypeScript port of comfyui-reshoot-nodes/reshoot_nodes/camera_json.py so the
// editor preview is exactly the path ComfyUI will render. OpenCV convention: x right, y down, z forward.
export type V3 = [number, number, number];
export type Keyframe = { t: number; pos: V3; look: V3 };
export type CameraPath = { version: 1; pivot_depth: number; ease: "smooth" | "linear"; freeze_time: boolean; keyframes: Keyframe[] };

export const MAX_ANGLE_DEG = 30;
export const MAX_TRAVEL = 0.5; // x pivot_depth

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3) => Math.hypot(a[0], a[1], a[2]);
const unit = (a: V3): V3 => mul(a, 1 / (norm(a) || 1));

export function originalCamera(pivot: number): Keyframe { return { t: 0, pos: [0, 0, 0], look: [0, 0, pivot] }; }

function catmull(ts: number[], pts: V3[], x: number): V3 {
  if (ts.length === 1) return [...pts[0]] as V3;
  let i = 0;
  while (i < ts.length - 2 && x >= ts[i + 1]) i++;
  const [t0, t1] = [ts[i], ts[i + 1]], [p0, p1] = [pts[i], pts[i + 1]];
  const m0 = i > 0 ? mul(sub(p1, pts[i - 1]), 1 / (t1 - ts[i - 1])) : mul(sub(p1, p0), 1 / (t1 - t0));
  const m1 = i + 2 < ts.length ? mul(sub(pts[i + 2], p0), 1 / (ts[i + 2] - t0)) : mul(sub(p1, p0), 1 / (t1 - t0));
  const h = t1 - t0, s = (x - t0) / h;
  const h00 = 2 * s ** 3 - 3 * s ** 2 + 1, h10 = s ** 3 - 2 * s ** 2 + s, h01 = -2 * s ** 3 + 3 * s ** 2, h11 = s ** 3 - s ** 2;
  return add(add(mul(p0, h00), mul(m0, h10 * h)), add(mul(p1, h01), mul(m1, h11 * h)));
}

export type Pose = { pos: V3; look: V3; angleDeg: number; travel: number; clamped: boolean };

export function clampPose(pos: V3, look: V3, pivot: number): Pose {
  let clamped = false;
  const lim = MAX_TRAVEL * pivot;
  const rawTravel = norm(pos);
  if (rawTravel > lim) { pos = mul(pos, lim / rawTravel); clamped = true; }
  let f = sub(look, pos);
  const dist = norm(f); f = unit(f);
  const f0: V3 = [0, 0, 1];
  const ang = Math.acos(Math.max(-1, Math.min(1, dot(f, f0))));
  const amax = (MAX_ANGLE_DEG * Math.PI) / 180;
  if (ang > amax + 1e-9) {
    const axis = unit(cross(f0, f)), c = Math.cos(amax), s = Math.sin(amax);
    f = add(add(mul(f0, c), mul(cross(axis, f0), s)), mul(axis, dot(axis, f0) * (1 - c)));
    clamped = true;
  }
  return { pos, look: add(pos, mul(f, dist)), angleDeg: (ang * 180) / Math.PI, travel: rawTravel / pivot, clamped };
}

/** Pose at normalised time u in [0,1] (ease applied, same as the ComfyUI node). */
export function poseAt(p: CameraPath, u: number): Pose {
  const k = [...p.keyframes].sort((a, b) => a.t - b.t);
  if (p.ease === "smooth") u = u * u * (3 - 2 * u);
  u = Math.min(u, k[k.length - 1].t);
  const ts = k.map((x) => x.t);
  return clampPose(catmull(ts, k.map((x) => x.pos), u), catmull(ts, k.map((x) => x.look), u), p.pivot_depth);
}

export function validate(p: CameraPath): string | null {
  const k = [...p.keyframes].sort((a, b) => a.t - b.t);
  if (!k.length) return "add at least one keyframe";
  if (k[0].t !== 0 || norm(k[0].pos) > 1e-6 || norm(sub(k[0].look, [0, 0, p.pivot_depth])) > 1e-6) return "keyframe 0 must be the original camera";
  for (let i = 1; i < k.length; i++) if (k[i].t <= k[i - 1].t) return "two keyframes share the same time";
  return null;
}
