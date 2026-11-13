"""P7-00: find which PDC-X test group leaves pytest alive after its final summary line (auditor-owned).

One pytest process at a time. A group is HANG when the summary line was printed but the process
did not exit within GRACE seconds (the process tree is then killed).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

GRACE = 30
LIMIT = 900
root = Path.cwd()
if len(sys.argv) > 1:  # optional: bisect the test files inside one directory
    groups = sorted((root / sys.argv[1]).glob("test_*.py"))
else:
    groups = sorted(p for p in (root / "tests/qualification/pdc_x").iterdir() if p.is_dir() and not p.name.startswith("__"))
SUMMARY = re.compile(r"\b\d+ (passed|failed|skipped|xfailed|error)\b.* in [\d.]+s")
results = []
for g in groups:
    rel = str(g.relative_to(root)).replace("\\", "/")
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-m", "pytest", rel, "-q", "--tb=no", "-p", "no:randomly"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=root,
    )
    state = {"summary_at": None, "tail": ""}

    def pump() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            if SUMMARY.search(line):
                state["summary_at"] = time.time()
                state["tail"] = line.strip()

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    status = None
    while status is None:
        if proc.poll() is not None:
            status = "EXITED"
        elif state["summary_at"] and time.time() - state["summary_at"] > GRACE:
            status = "HANG_AFTER_SUMMARY"
        elif time.time() - t0 > LIMIT:
            status = "TIMEOUT"
        else:
            time.sleep(1)
    if status != "EXITED":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    th.join(timeout=5)
    rec = {"group": rel, "status": status, "exit": proc.poll(), "seconds": round(time.time() - t0, 1), "summary": state["tail"]}
    results.append(rec)
    print(json.dumps(rec), flush=True)
print("DONE", json.dumps([r["group"] for r in results if r["status"] != "EXITED"]))
