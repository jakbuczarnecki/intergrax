"""P7-00 addendum 4: ONE watched pytest process (auditor-owned).

Usage: python addendum_4_guarded_pytest.py <checkout_root> <log> <limit_seconds> <grace_seconds> <pytest args...>
Writes the pytest output to <log>. Records whether the process exited by itself after its final summary line (F-09 exit
proof); if it is still alive <grace_seconds> after the summary it is killed and the status says so.
"""

import re
import subprocess
import sys
import threading
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
log, limit, grace = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
args = sys.argv[5:]
S = re.compile(r"\b\d+ (passed|failed|skipped|xfailed|error)\b.* in [\d.]+s")
proc = subprocess.Popen(
    [sys.executable, "-m", "pytest", *args], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)
st: dict[str, float | None] = {"t": None}
out = open(log, "w", encoding="utf-8")


def pump() -> None:
    for line in proc.stdout:  # type: ignore[union-attr]
        out.write(line)
        out.flush()
        if S.search(line):
            st["t"] = time.time()


th = threading.Thread(target=pump, daemon=True)
th.start()
t0 = time.time()
status = None
while status is None:
    if proc.poll() is not None:
        status = "EXITED_BY_ITSELF"
    elif st["t"] and time.time() - st["t"] > grace:
        status = f"STILL_ALIVE_{grace}S_AFTER_SUMMARY_KILLED"
    elif time.time() - t0 > limit:
        status = "TIMEOUT_KILLED"
    else:
        time.sleep(1)
if status != "EXITED_BY_ITSELF":
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
th.join(timeout=10)
tsum = (st["t"] or t0) - t0
out.write(f"RUNNER status={status} exit={proc.poll()} seconds_to_summary={tsum:.1f} total={time.time() - t0:.1f}\n")
out.close()
print(status)
