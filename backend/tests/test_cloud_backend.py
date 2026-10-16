"""Comfy Cloud backend with mocked HTTP (no network, no real key)."""
import json
import pytest
from reshoot.backends import CloudComfyBackend, make_backend


class Rec(CloudComfyBackend):
    def __init__(self, replies):
        super().__init__("test-key-123", "https://cloud.example")
        self.replies, self.calls = replies, []

    def _raw(self, method, path, data, headers):
        self.calls.append((method, path, headers))
        return self.replies[path] if isinstance(self.replies[path], bytes) else json.dumps(self.replies[path]).encode()


def _patch(monkeypatch, b):
    import reshoot.backends as m
    monkeypatch.setattr(m.NativeComfyBackend, "_http", lambda self, me, p, d=None, h=None: self._raw(me, p, d, h))


def test_make_backend_requires_key(monkeypatch):
    monkeypatch.delenv("COMFY_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        make_backend("cloud")
    monkeypatch.setenv("COMFY_API_KEY", "sekrit-zz9")
    monkeypatch.setenv("COMFY_BASE_URL", "https://cloud.example/")
    b = make_backend("cloud")
    assert b.base == "https://cloud.example/api" and "sekrit-zz9" not in repr(b)


def test_key_header_jobs_and_view(monkeypatch):
    b = Rec({"/prompt": {"prompt_id": "p1"},
             "/jobs/p1": {"id": "p1", "status": "completed",
                          "outputs": {"90": {"gifs": [{"filename": "o.mp4", "subfolder": "", "type": "output"}]}}},
             "/view?filename=o.mp4&subfolder=&type=output": b"MP4"})
    _patch(monkeypatch, b)
    assert b.run_prompt({"1": {}}, "90") == [b"MP4"]
    assert all(h.get("X-API-Key") == "test-key-123" for _, _, h in b.calls)
    assert [p for _, p, _ in b.calls][0] == "/prompt"


def test_failed_job_raises(monkeypatch):
    b = Rec({"/prompt": {"prompt_id": "p2"},
             "/jobs/p2": {"id": "p2", "status": "failed", "execution_error": {"exception_message": "boom"}}})
    _patch(monkeypatch, b)
    with pytest.raises(RuntimeError, match="boom"):
        b.run_prompt({"1": {}}, "90")


def test_balance_and_patch_graph(monkeypatch):
    b = Rec({"/billing/usage/timeseries": {"summary": {"balance": {"amount_cents": 1984}}}})
    _patch(monkeypatch, b)
    assert b.balance_cents() == 1984
    g = b.patch_graph({"30": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "x.gguf"}},
                       "34": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {"lora_name": "local", "model": ["30", 0]}}})
    assert g["30"]["class_type"] == "UNETLoader" and g["30"]["inputs"]["unet_name"] == b.UNET
    assert g["34"]["inputs"]["lora_name"] == b.LORA
    dg, out = b.depth_graph("v.mp4", type("O", (), {"fps": 24, "width": 448, "height": 448, "frames": 33})())
    assert out in dg and any(n["class_type"] == "DepthAnythingV2Preprocessor" for n in dg.values())
    assert b.free() is None and b.preflight() == {"cloud": True}
