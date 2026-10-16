import os, subprocess, time, tempfile
from pathlib import Path
import pytest

os.environ["RESHOOT_BACKEND"] = "mock"
os.environ["RESHOOT_DATA"] = tempfile.mkdtemp()
from fastapi.testclient import TestClient  # noqa: E402
from reshoot.main import app  # noqa: E402
from reshoot.presets import clamp_strength  # noqa: E402

c = TestClient(app)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "in.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=12:duration=1",
                    "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def test_clamp():
    assert clamp_strength(90) == 30 and clamp_strength(-45) == -30 and clamp_strength(10) == 10


def test_presets():
    r = c.get("/api/presets").json()
    assert {p["id"] for p in r["presets"]} >= {"orbit_left", "orbit_right", "low", "high", "push", "pull", "truck", "crane"}


def test_rejects_bad_upload():
    assert c.post("/api/uploads", files={"file": ("x.exe", b"MZ")}).status_code == 400


def test_end_to_end(clip):
    up = c.post("/api/uploads", files={"file": ("in.mp4", clip.read_bytes(), "video/mp4")}).json()
    presets = ["orbit_left", "orbit_right", "low", "high", "push", "pull", "truck", "crane"]
    job = c.post("/api/jobs", json={"upload_id": up["upload_id"],
                                    "angles": [{"preset": p, "angle_deg": 40} for p in presets]}).json()
    assert all(a["angle_deg"] == 30 for a in job["angles"])
    for _ in range(120):
        j = c.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert j["status"] == "done", j
    for a in j["angles"]:
        assert a["status"] == "done", a
        r = c.get(f"/api/outputs/{a['output']}")
        assert r.status_code == 200 and len(r.content) > 1000


def test_unknown_preset(clip):
    up = c.post("/api/uploads", files={"file": ("in.mp4", clip.read_bytes())}).json()
    assert c.post("/api/jobs", json={"upload_id": up["upload_id"], "angles": [{"preset": "barrel_roll"}]}).status_code == 400


def test_upload_notice_and_limits(clip):
    r = c.post("/api/uploads", files={"file": ("in.mp4", clip.read_bytes())}).json()
    assert "trimmed" in r["notice"] and "640x352" in r["notice"]
    assert c.get("/api/limits").json()["frames"] == 41


def _wait(job_id):
    for _ in range(120):
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.5)


def test_depth_preview_and_custom_path(clip):
    up = c.post("/api/uploads", files={"file": ("in.mp4", clip.read_bytes())}).json()
    d = c.get(f"/api/uploads/{up['upload_id']}/depth").json()
    import base64
    assert len(base64.b64decode(d["depth"])) == d["grid_w"] * d["grid_h"]
    assert len(base64.b64decode(d["rgb"])) == 3 * d["grid_w"] * d["grid_h"]
    assert d["hfov_deg"] == 60 and d["source"] == "heuristic"
    seed = c.get("/api/presets/orbit_left/path?angle_deg=20").json()
    assert seed["keyframes"][0]["pos"] == [0, 0, 0]
    j = c.post("/api/jobs", json={"upload_id": up["upload_id"], "angles": [{"preset": "custom", "path": seed}]}).json()
    j = _wait(j["id"])
    assert j["status"] == "done" and j["angles"][0]["path"]


def test_custom_path_rejected_when_invalid(clip):
    up = c.post("/api/uploads", files={"file": ("in.mp4", clip.read_bytes())}).json()
    bad = {"keyframes": [{"t": 0, "pos": [1, 0, 0], "look": [0, 0, 3]}]}
    r = c.post("/api/jobs", json={"upload_id": up["upload_id"], "angles": [{"preset": "custom", "path": bad}]})
    assert r.status_code == 400 and "keyframe 0" in r.json()["detail"]


# ---------- hardware tiers, chunked jobs, cancel/resume (mock backend emulates the 16 GB tier)
@pytest.fixture(scope="module")
def long_clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "long.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=6",
                    "-f", "lavfi", "-i", "sine=duration=6", "-shortest", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def _wait(jid, until=("done", "error", "cancelled"), t=60):
    for _ in range(t * 10):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in until:
            return j
        time.sleep(0.1)
    raise AssertionError(j)


def test_hardware_endpoint():
    h = c.get("/api/hardware").json()
    assert h["tier"]["id"] == "16gb" and h["caps"]["max_seconds"] == 5 and h["caps"]["resolutions"] == ["640x352"]


def test_tier_limits_enforced(long_clip):
    up = c.post("/api/uploads", files={"file": ("l.mp4", long_clip.read_bytes(), "video/mp4")}).json()
    base = {"upload_id": up["upload_id"], "angles": [{"preset": "orbit_left", "angle_deg": 15}]}
    assert c.post("/api/jobs", json={**base, "duration_s": 8}).status_code == 400          # > 5 s on 16 GB
    assert c.post("/api/jobs", json={**base, "resolution": "960x544"}).status_code == 400
    assert c.post("/api/jobs", json={**base, "duration_s": 3, "upscale": True}).status_code == 400   # > 25 frames
    j = c.post("/api/jobs", json={**base, "duration_s": 5}).json()
    assert j["opts"]["chunked"] and j["opts"]["frames"] == 120 and j["opts"]["window"] == 33
    j = _wait(j["id"])
    assert j["status"] == "done" and j["angles"][0]["stats"]["chunks"] == 5
    j = c.post("/api/jobs", json={**base, "duration_s": 1, "upscale": True}).json()
    assert j["opts"]["upscale"] and j["opts"]["frames"] == 17
    assert _wait(j["id"])["status"] == "done"


def test_cancel_and_resume(long_clip, monkeypatch):
    from reshoot import main
    up = c.post("/api/uploads", files={"file": ("l.mp4", long_clip.read_bytes(), "video/mp4")}).json()
    orig = main.queue.backend.render

    def slow(*a, **k):
        for _ in range(100):
            if k["cancelled"]():
                from reshoot.chunked import Cancelled
                raise Cancelled("x")
            time.sleep(0.02)
        return orig(*a, **k)
    monkeypatch.setattr(main.queue.backend, "render", slow)
    j = c.post("/api/jobs", json={"upload_id": up["upload_id"], "duration_s": 5,
                                  "angles": [{"preset": "push", "angle_deg": 10}]}).json()
    time.sleep(0.2)
    c.post(f"/api/jobs/{j['id']}/cancel")
    j = _wait(j["id"])
    assert j["status"] == "cancelled" and j["angles"][0]["status"] == "cancelled"
    monkeypatch.setattr(main.queue.backend, "render", orig)
    c.post(f"/api/jobs/{j['id']}/resume")
    assert _wait(j["id"])["status"] == "done"


def test_jobs_reload_after_restart(tmp_path):
    from reshoot.jobs import JobQueue, Angle, Job
    from reshoot.backends import MockBackend
    q = JobQueue(tmp_path, MockBackend())
    j = Job("abc123", "u1", [Angle("push", 10.0, status="running")], status="running", opts={"frames": 120})
    q.jobs[j.id] = j; q._save(j)
    q2 = JobQueue(tmp_path, MockBackend())
    r = q2.get("abc123")
    assert r.status == "error" and r.angles[0].status == "error" and r.opts["frames"] == 120


def test_social_export_endpoint(clip):
    up = c.post("/api/uploads", files={"file": ("in.mp4", clip.read_bytes(), "video/mp4")}).json()
    job = c.post("/api/jobs", json={"upload_id": up["upload_id"], "angles": [{"preset": "orbit_left"}]}).json()
    for _ in range(120):
        j = c.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert j["status"] == "done", j
    assert c.post(f"/api/jobs/{j['id']}/social", json={"layout": "diagonal"}).status_code == 400
    assert c.post(f"/api/jobs/{j['id']}/social", json={"angle": 5}).status_code == 400
    assert c.post("/api/jobs/nope/social", json={}).status_code == 404
    r = c.post(f"/api/jobs/{j['id']}/social", json={"layout": "stacked", "labels": ["Original", "Re-shot"],
                                                    "credit": "Made with ReAngle · ComfyUI + LTX"})
    assert r.status_code == 200, r.text
    v = c.get(r.json()["url"])
    assert v.status_code == 200 and len(v.content) > 10000
