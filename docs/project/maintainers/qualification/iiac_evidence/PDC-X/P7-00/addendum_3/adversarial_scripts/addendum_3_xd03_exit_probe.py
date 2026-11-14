"""P7-00 addendum 3: status of F-09 (pytest does not exit after test_xd_03_agent_governance.py). One pytest process,
watched; killed 90 s after the final summary line if still alive. Usage: python <this> <checkout_root> <log>"""
import re, subprocess, sys, threading, time
from pathlib import Path
root = Path(sys.argv[1]).resolve()
S = re.compile(r"\b\d+ (passed|failed|skipped|xfailed|error)\b.* in [\d.]+s")
proc = subprocess.Popen([sys.executable, "-m", "pytest", "tests/qualification/pdc_x/xd/test_xd_03_agent_governance.py", "-q", "--tb=short"],
                        cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
st = {"t": None}; out = open(sys.argv[2], "w", encoding="utf-8")
def pump():
    for line in proc.stdout:
        out.write(line); out.flush()
        if S.search(line): st["t"] = time.time()
th = threading.Thread(target=pump, daemon=True); th.start()
t0 = time.time(); status = None
while status is None:
    if proc.poll() is not None: status = "EXITED_BY_ITSELF"
    elif st["t"] and time.time() - st["t"] > 90: status = "STILL_ALIVE_90S_AFTER_SUMMARY_KILLED"
    elif time.time() - t0 > 900: status = "TIMEOUT"
    else: time.sleep(1)
if status != "EXITED_BY_ITSELF":
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
th.join(timeout=10)
out.write(f"RUNNER status={status} exit={proc.poll()} seconds_to_summary={(st['t'] or t0)-t0:.1f} total={time.time()-t0:.1f}\n"); out.close()
print(status)
