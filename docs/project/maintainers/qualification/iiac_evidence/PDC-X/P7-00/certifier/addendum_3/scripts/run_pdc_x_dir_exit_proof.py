"""F-09 / F-41 / F-49 / F-50 status: the whole tests/qualification/pdc_x directory in ONE pytest process,
without os._exit, so that the process has to exit by itself (the auditor's F-09 claim). A wall-clock cap kills it if not."""
import os, subprocess, sys, time
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print("# certifier pdc_x directory run (PDC-P7-00 preflight addendum 3)")
print("# command: python -m pytest tests/qualification/pdc_x -q -rf --tb=line   (child of this script, cap 1500 s)")
print("# HEAD:", sha, "| checkout:", os.getcwd(), "| PYTHONPATH:", os.environ.get("PYTHONPATH")); sys.stdout.flush()
t = time.time()
p = subprocess.Popen([sys.executable, "-m", "pytest", "tests/qualification/pdc_x", "-q", "-rf", "--tb=line"])
try:
    rc = p.wait(timeout=1500)
    print("# pytest process exited by itself, exit code:", rc)
except subprocess.TimeoutExpired:
    p.kill(); rc = -9
    print("# pytest process still alive after 1500 s: killed (F-09 behaviour)")
print("# elapsed_s: %.2f" % (time.time() - t)); sys.stdout.flush()
os._exit(0 if rc == 0 else 1)
