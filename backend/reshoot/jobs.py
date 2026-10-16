"""Tiny in-process job queue with JSON persistence."""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Queue

from .backends import RenderBackend


@dataclass
class Angle:
    preset: str
    angle_deg: float
    path: str | None = None          # custom camera path JSON (preset == "custom")
    status: str = "queued"
    progress: float = 0.0
    output: str | None = None
    error: str | None = None
    detail: str | None = None        # e.g. "Chunk 2/4 · 95 s · GPU 12100 MB peak"
    stats: dict | None = None        # timings / peak VRAM per chunk


@dataclass
class Job:
    id: str
    upload_id: str
    angles: list[Angle]
    status: str = "queued"       # queued | running | done | error | cancelled
    created: float = field(default_factory=time.time)
    backend: str = ""
    opts: dict = field(default_factory=dict)   # tier-validated: frames, width, height, upscale, window, ...
    cancel: bool = False

    @property
    def progress(self) -> float:
        return sum(a.progress for a in self.angles) / max(len(self.angles), 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["progress"] = round(self.progress, 3)
        return d


class JobQueue:
    def __init__(self, data_dir: Path, backend: RenderBackend):
        self.data_dir, self.backend = data_dir, backend
        self.uploads, self.outputs = data_dir / "uploads", data_dir / "outputs"
        for d in (self.uploads, self.outputs, data_dir / "jobs"):
            d.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.q: Queue[str] = Queue()
        self.lock = threading.Lock()
        self._load()
        threading.Thread(target=self._worker, daemon=True).start()

    def _load(self) -> None:
        """Reload saved jobs so a failed/cancelled render can be resumed after an app restart."""
        for f in sorted((self.data_dir / "jobs").glob("*.json")):
            try:
                d = json.loads(f.read_text())
                d.pop("progress", None)
                angles = [Angle(**a) for a in d.pop("angles")]
                job = Job(angles=angles, **{k: v for k, v in d.items() if k in Job.__dataclass_fields__})
            except Exception:  # noqa: BLE001
                continue
            if job.status in ("queued", "running"):
                job.status = "error"
                for a in job.angles:
                    if a.status in ("queued", "running"):
                        a.status, a.error = "error", "app restarted during render; press Resume"
            self.jobs[job.id] = job

    def upload_path(self, upload_id: str) -> Path | None:
        hits = [h for h in self.uploads.glob(f"{upload_id}.*") if not h.name.endswith(".json")]
        return hits[0] if hits else None

    def submit(self, upload_id: str, angles: list[tuple], opts: dict | None = None) -> Job:
        job = Job(uuid.uuid4().hex[:12], upload_id, [Angle(*a) for a in angles], backend=self.backend.name,
                  opts=opts or {})
        with self.lock:
            self.jobs[job.id] = job
        self._save(job)
        self.q.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job and job.status in ("queued", "running"):
            job.cancel = True
        return job

    def resume(self, job_id: str) -> Job | None:
        """Re-run failed/cancelled angles; finished chunks in their work dirs are reused."""
        job = self.jobs.get(job_id)
        if not job or job.status in ("queued", "running"):
            return job
        for a in job.angles:
            if a.status in ("error", "cancelled"):
                a.status, a.error, a.progress, a.detail = "queued", None, 0.0, "resuming"
        job.cancel, job.status = False, "queued"
        self._save(job)
        self.q.put(job.id)
        return job

    def _save(self, job: Job) -> None:
        (self.data_dir / "jobs" / f"{job.id}.json").write_text(json.dumps(job.to_dict(), indent=2))

    def _worker(self) -> None:
        while True:
            job = self.jobs[self.q.get()]
            job.status = "running"
            src = self.upload_path(job.upload_id)
            for i, ang in enumerate(job.angles):
                if ang.status == "done":
                    continue
                if job.cancel:
                    ang.status, ang.detail = "cancelled", "cancelled"
                    continue
                ang.status = "running"
                out = self.outputs / f"{job.id}_{i}_{ang.preset}.mp4"
                work = self.data_dir / "jobs" / f"{job.id}_{i}"

                def cb(p: float, detail: str | None = None, a=ang) -> None:
                    a.progress = p
                    if detail:
                        a.detail = detail
                try:
                    if src is None:
                        raise FileNotFoundError("upload missing")
                    self.backend.render(src, ang.preset, ang.angle_deg, out, cb, path=ang.path, opts=job.opts,
                                        work=work, cancelled=lambda j=job: j.cancel)
                    ang.status, ang.output, ang.progress = "done", out.name, 1.0
                    ang.stats = getattr(self.backend, "last_stats", None)
                    ang.detail = None
                except Exception as e:  # noqa: BLE001
                    if type(e).__name__ == "Cancelled" or job.cancel:
                        ang.status, ang.detail = "cancelled", "cancelled; finished chunks are kept, press Resume"
                    else:
                        ang.status, ang.error = "error", f"{e}"
                        traceback.print_exc()
                self._save(job)
            st = [a.status for a in job.angles]
            job.status = "cancelled" if "cancelled" in st else ("error" if all(x == "error" for x in st) else "done")
            self._save(job)
