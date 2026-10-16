"""ComfyBackend wiring test with a fake SDK client (no network, no key)."""
from pathlib import Path
from reshoot.backends import ComfyBackend, make_backend
import pytest


class FakeOut:
    content_type = "video/mp4"
    def to_file(self, p): Path(p).write_bytes(b"x" * 2000)


class FakeJob:
    outputs = [FakeOut()]
    def result(self): return self


class FakeWF:
    def __init__(self, g): self.g, self.set = g, []
    def set_input(self, n, f, v): assert n in self.g; self.set.append((n, f, v))


class FakeClient:
    last = None
    def __enter__(self): return self
    def __exit__(self, *a): pass
    class workflows:
        @staticmethod
        def from_json(g): FakeClient.last = FakeWF(g); return FakeClient.last
    class assets:
        @staticmethod
        def from_file(p): return ("ASSET", str(p))
    def submit(self, wf, metadata=None): return FakeJob()


def test_comfy_backend_binds_inputs(tmp_path, monkeypatch):
    b = ComfyBackend("comfy", None, "k")
    monkeypatch.setattr(b, "_client", lambda: FakeClient())
    src = tmp_path / "in.mp4"; src.write_bytes(b"v")
    out = b.render(src, "low", 12.0, tmp_path / "o.mp4", lambda p: None)
    assert out.stat().st_size == 2000
    assert ("20", "preset", "low") in FakeClient.last.set and ("20", "angle_deg", 12.0) in FakeClient.last.set
    assert FakeClient.last.set[0][2][0] == "ASSET"


def test_comfy_requires_key(monkeypatch):
    monkeypatch.delenv("COMFY_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        make_backend("comfy")
