"""Submit a ReAngle API workflow to a plain ComfyUI and report time, VRAM peak and outputs.

python comfy_run.py workflow.json --set video=clip.mp4 --set angle_deg=20 [--host 127.0.0.1:8001]
Stdlib only, so it runs with ComfyUI's own venv python.
"""
import argparse, json, subprocess, threading, time, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("workflow")
ap.add_argument("--host", default="127.0.0.1:8001")
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--timeout", type=float, default=3600)
a = ap.parse_args()
raw = json.load(open(a.workflow, encoding="utf-8"))
bind = raw.pop("_reshoot_bindings", {})
raw.pop("_comment", None)
for kv in a.set:
    k, v = kv.split("=", 1)
    node, field = bind[k]
    try:
        v = json.loads(v)
    except ValueError:
        pass
    raw[node]["inputs"][field] = v

peak = {"mib": 0, "base": None}
stop = False
def smi():
    while not stop:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
            m = int(out); peak["base"] = m if peak["base"] is None else peak["base"]; peak["mib"] = max(peak["mib"], m)
        except Exception:
            pass
        time.sleep(1)
threading.Thread(target=smi, daemon=True).start()

def req(path, data=None):
    r = urllib.request.Request(f"http://{a.host}{path}", data=json.dumps(data).encode() if data is not None else None,
                               headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(r, timeout=60))
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:4000]); raise SystemExit(2)

t0 = time.time()
pid = req("/prompt", {"prompt": raw, "client_id": uuid.uuid4().hex})["prompt_id"]
print("prompt_id", pid, flush=True)
while time.time() - t0 < a.timeout:
    h = req(f"/history/{pid}")
    if pid in h:
        st = h[pid].get("status", {})
        if st.get("completed") or st.get("status_str") in ("success", "error"):
            break
    time.sleep(2)
stop = True
dt = time.time() - t0
st = h[pid]["status"]
print(json.dumps({"status": st.get("status_str"), "seconds": round(dt, 1), "vram_peak_mib": peak["mib"],
                  "vram_start_mib": peak["base"]}))
for m in st.get("messages", []):
    if m[0] in ("execution_error", "execution_interrupted"):
        d = m[1]; print("ERROR node", d.get("node_id"), d.get("node_type"), d.get("exception_message", "")[:3000])
        print("".join(d.get("traceback", []))[-3000:])
    if m[0] == "execution_cached": print("cached:", m[1].get("nodes"))
for nid, o in h[pid].get("outputs", {}).items():
    for key in ("gifs", "videos", "images"):
        for f in o.get(key, []):
            print("OUTPUT", nid, f.get("subfolder"), f.get("filename"), f.get("fullpath", ""))
