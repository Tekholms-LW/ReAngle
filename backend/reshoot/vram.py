"""Fast VRAM sampler: one long-lived `nvidia-smi -lms` process streaming readings every ~200 ms.

The old watchdog read nvidia-smi every 2 s and needed two consecutive readings over the limit, so a short
spike (e.g. 15,846 / 16,303 MB during VAE decode) could slip through. This samples ~10x faster, keeps the true
peak, and applies two limits:
  * hard: any single reading above total - RESHOOT_VRAM_HARD_MARGIN_MB (default 300) -> stop now
  * soft: readings above total - RESHOOT_VRAM_ABORT_MARGIN_MB (default 450) for >= RESHOOT_VRAM_SOFT_S (1.0 s)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time


class VramSampler:
    def __init__(self, interval_ms: int = 200, strikes: int = 0):
        self.strikes, self._strk = strikes, 0       # strikes>0: count consecutive slow readings instead of seconds
        self.interval_ms = interval_ms
        self.peak = 0
        self.total = 0
        self.last = None
        self.n = 0
        self.hard = int(os.environ.get("RESHOOT_VRAM_HARD_MARGIN_MB", "300"))
        self.soft = int(os.environ.get("RESHOOT_VRAM_ABORT_MARGIN_MB", "450"))
        self.soft_s = float(os.environ.get("RESHOOT_VRAM_SOFT_S", "1.0"))
        self.trip: str | None = None
        self._over_since: float | None = None
        self._p = None
        self._t = None

    def start(self) -> bool:
        exe = shutil.which("nvidia-smi")
        if not exe:
            return False
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._p = subprocess.Popen([exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits",
                                    f"-lms={self.interval_ms}"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   text=True, creationflags=flags)
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()
        return True

    def feed(self, used: int, total: int, now: float | None = None) -> None:
        """One reading (also used by tests)."""
        if total <= 0 or not (0 <= used <= total * 1.05):        # WDDM garbage
            return
        now = time.time() if now is None else now
        self.n += 1
        self.total, self.last = total, used
        self.peak = max(self.peak, used)
        if self.trip:
            return
        if used > total - self.hard:
            self.trip = f"GPU memory spiked to {used} of {total} MB (hard margin {self.hard} MB)"
        elif used > total - self.soft and self.strikes:
            self._strk += 1
            if self._strk >= self.strikes:
                self.trip = (f"GPU memory stayed above {total - self.soft} MB for {self.strikes} readings "
                             f"(reached {used} of {total} MB)")
        elif used > total - self.soft:
            if self._over_since is None:
                self._over_since = now
            if now - self._over_since >= self.soft_s:
                self.trip = (f"GPU memory stayed above {total - self.soft} MB for {self.soft_s:g} s "
                             f"(reached {used} of {total} MB)")
        else:
            self._over_since, self._strk = None, 0

    def _read(self):
        for line in self._p.stdout:
            try:
                u, t = (int(x.strip()) for x in line.split(","))
            except ValueError:
                continue
            self.feed(u, t)

    def stop(self):
        if self._p and self._p.poll() is None:
            self._p.terminate()
            try:
                self._p.wait(3)
            except Exception:  # noqa: BLE001
                self._p.kill()
