import os; os.environ.setdefault("RESHOOT_PREFLIGHT_WAIT_S", "0")
"""NativeComfyBackend against a fake in-process ComfyUI HTTP server."""
import json, subprocess, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from reshoot.backends import NativeComfyBackend

STATE = {}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj, ctype="application/json"):
        b = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", ctype); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/free":
            STATE["freed"] = True; self._send({})
        elif self.path == "/upload/image":
            i = body.find(b"\r\n\r\n", body.find(b'name="image"')) + 4
            STATE["upload"] = body[i:body.rfind(b"\r\n--")]
            self._send({"name": "up.mp4", "subfolder": "", "type": "input"})
        else:
            STATE["prompt"] = json.loads(body)["prompt"]; self._send({"prompt_id": "p1", "node_errors": {}})
    def do_GET(self):
        if self.path == "/queue":
            self._send({"queue_running": STATE.get("busy", []), "queue_pending": []})
        elif self.path == "/system_stats":
            self._send({"system": {"argv": STATE.get("argv", ["main.py", "--reserve-vram", "3"])}, "devices": [{"vram_free": STATE.get("free_mb", 14000) << 20, "vram_total": 16303 << 20}]})
        elif self.path.startswith("/history"):
            self._send({"p1": {"status": {"completed": True, "status_str": "success", "messages": []},
                               "outputs": {"90": {"gifs": [{"filename": "r.mp4", "subfolder": "reangle", "type": "output"}]}}}})
        else:
            STATE["view"] = self.path; self._send(b"x" * 3000, "video/mp4")


def test_native_backend_roundtrip(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    src = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=24:duration=10",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    b = NativeComfyBackend(f"http://127.0.0.1:{srv.server_port}", max_frames=41)
    b_render = b.render(src, "orbit_right", 25.0, tmp_path / "o.mp4", lambda p: None)
    p = STATE["prompt"]
    assert p["1"]["inputs"]["video"] == "up.mp4" and p["1"]["inputs"]["frame_load_cap"] == 41
    assert p["20"]["inputs"]["preset"] == "orbit_right" and p["20"]["inputs"]["angle_deg"] == 25.0
    assert STATE.get("freed")
    assert "subfolder=reangle" in STATE["view"] and b_render.stat().st_size == 3000
    up = tmp_path / "uploaded.mp4"; up.write_bytes(STATE["upload"])
    from reshoot.backends import probe
    info = probe(up)
    assert (info["width"], info["height"], info["frames"]) == (640, 352, 41)
    srv.shutdown()


def _server():
    srv = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_preflight_refuses_low_vram_and_busy(tmp_path):
    import pytest
    srv = _server(); b = NativeComfyBackend(f"http://127.0.0.1:{srv.server_port}")
    STATE["free_mb"] = 4000
    with pytest.raises(RuntimeError, match="GPU memory"):
        b.preflight()
    STATE["free_mb"] = 14000; STATE["busy"] = [["x"]]
    with pytest.raises(RuntimeError, match="busy"):
        b.preflight()
    STATE["busy"] = []
    STATE["argv"] = ["main.py", "--port", "8001"]
    with pytest.raises(RuntimeError, match="reserve-vram"):
        b.preflight()
    STATE["argv"] = ["main.py", "--reserve-vram", "3"]
    assert b.preflight()["vram_free_mb"] == 14000
    srv.shutdown()


def test_frames_are_8n_plus_1(tmp_path):
    src = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24:duration=1",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    assert NativeComfyBackend("http://x", max_frames=97).frames_for(src) == 17


def test_native_backend_custom_path(tmp_path):
    srv = _server()
    src = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=2",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    from reshoot.camerapath import preset_to_json
    path = json.dumps(preset_to_json("crane", 15))
    NativeComfyBackend(f"http://127.0.0.1:{srv.server_port}").render(src, "custom", 0.0, tmp_path / "o.mp4",
                                                                      lambda p: None, path=path)
    p = STATE["prompt"]
    assert "20" not in p and p["21"]["inputs"]["camera_path"] == ["22", 0]
    assert json.loads(p["22"]["inputs"]["path_json"])["keyframes"][0]["pos"] == [0.0, 0.0, 0.0]
    assert p["22"]["inputs"]["num_frames"] == 41
    srv.shutdown()


def test_native_backend_preset_drops_json_node(tmp_path):
    srv = _server()
    src = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=2",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    NativeComfyBackend(f"http://127.0.0.1:{srv.server_port}").render(src, "push", 10.0, tmp_path / "o.mp4", lambda p: None)
    p = STATE["prompt"]
    assert "22" not in p and p["21"]["inputs"]["camera_path"] == ["20", 0] and p["20"]["inputs"]["preset"] == "push"
    srv.shutdown()


def test_vram_watchdog_ignores_garbage_and_aborts_on_real_overrun(monkeypatch):
    import json as _j
    from reshoot.backends import NativeComfyBackend
    b = NativeComfyBackend("http://127.0.0.1:1")
    calls = []

    def http(method, path, data=None, headers=None):
        calls.append(path)
        if path == "/prompt":
            return _j.dumps({"prompt_id": "p"}).encode()
        if path.startswith("/history"):
            return _j.dumps({}).encode()
        return b"{}"
    monkeypatch.setattr(b, "_http", http)
    monkeypatch.setattr("time.sleep", lambda s: None)
    seq = iter([(17592181866204, 16303), (9000, 16303), (15900, 16303), (15950, 16303)])
    monkeypatch.setattr(b, "vram_used_mb", lambda: next(seq))
    import pytest
    with pytest.raises(RuntimeError, match="15950 of 16303"):
        b.run_prompt({}, "90")
    assert "/interrupt" in calls and b.last_peak_mb == 15950
