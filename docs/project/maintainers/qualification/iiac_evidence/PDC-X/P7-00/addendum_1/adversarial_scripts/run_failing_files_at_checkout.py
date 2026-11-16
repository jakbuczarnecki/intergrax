"""P7-00 addendum 1: re-run the files that failed in the regression run in a short-path detached checkout.

Usage: python run_failing_files_at_checkout.py <checkout_root> <log_path> <main_run_stdout_log>
One pytest process; killed 30 s after the final summary line (exit-hang guard).
"""
import os, re, subprocess, sys, threading, time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
FILES = [
    "tests/unit/agent_distribution/test_agent_distribution_materialization_adapter.py",
    "tests/unit/agent_distribution/test_agent_distribution_materialization.py",
    "tests/unit/agent_distribution/test_ac6_architecture_gates.py",
    "tests/unit/agent_distribution/test_agent_distribution_build_remediation.py",
    "tests/unit/agent_distribution/test_agent_distribution_activation_remediation.py",
    "tests/qualification/compat_x/test_compat_x_event_evolution_r3_r1_gates.py",
    "tests/qualification/compat_x/test_compat_x_event_evolution_r3_p0_gates.py",
    "tests/qualification/compat_x/test_compat_x_versioning_r1_r3_identity_gates.py",
    "tests/qualification/compat_x/test_compat_x_versioning_r1_r5_semantic_identity_reconciliation_gates.py",
    "tests/qualification/compat_x/test_compat_x_deprecation_r5_p0_gates.py",
    "tests/unit/governance/test_pdc_x_inventory.py",
]
# node ids of the failures of the main regression run (argv[3] = its stdout log); only existing files are passed
failed = []
for line in open(sys.argv[3], encoding="utf-8", errors="replace"):
    if line.startswith(("FAILED ", "ERROR ")):
        failed.append(line.split(" ", 1)[1].split(" - ")[0].strip())
files = [n for n in dict.fromkeys(failed) if (root / n.split("::")[0]).is_file()]
env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), os.environ.get("EXTRA_PP", "")]))
S = re.compile(r"\b\d+ (passed|failed|skipped|xfailed|error)\b.* in [\d.]+s")
proc = subprocess.Popen([sys.executable, "-m", "pytest", *files, "-q", "-rf", "--tb=line"], cwd=root, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
st = {"t": None}
out = open(sys.argv[2], "w", encoding="utf-8")
def pump():
    for line in proc.stdout:
        out.write(line); out.flush()
        if S.search(line): st["t"] = time.time()
th = threading.Thread(target=pump, daemon=True); th.start()
t0 = time.time()
while proc.poll() is None:
    if st["t"] and time.time() - st["t"] > 30: break
    if time.time() - t0 > 1500: break
    time.sleep(1)
if proc.poll() is None:
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
th.join(timeout=10)
out.write(f"RUNNER files={len(files)} elapsed={time.time()-t0:.1f}s\n"); out.close()
