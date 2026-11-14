"""P7-00 addendum 1: ONE bounded pytest process for the regression run (auditor-owned).

Contents: MCP-RC mandatory contract (list from Python), all PDC-X tests, the EBH-5 gate files, the PDC-X governance
tests (inventory, requirements, WBS), unit memory and agent_distribution suites, COMPAT-X qualification, agent-contract
unit tests. The process is killed GRACE seconds after pytest printed its final summary line (known pytest exit hang
after tests/qualification/pdc_x/xd/test_xd_03_agent_governance.py, finding F-09 of the first audit).
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from tests.qualification.mcp_server_x.rc_selection import MCP_SERVER_X_RC_SELECTION  # noqa: E402

GRACE = 30
LIMIT = 3600
root = Path.cwd()


def rel(paths):
    return sorted({str(p.relative_to(root)).replace("\\", "/") for p in paths})


files = list(MCP_SERVER_X_RC_SELECTION)
files += rel((root / "tests/qualification/pdc_x").rglob("test_*.py"))
files += [
    "tests/qualification/ebh_5/test_ebh_5_g06_enterprise_integrity_gates.py",
    "tests/qualification/ebh_5/test_ebh_5_proof_integrity_gates.py",
    "tests/qualification/ebh_5/test_ebh_5_rpl_f_integrity_gates.py",
    "tests/unit/governance/test_pdc_x_inventory.py",
    "tests/unit/governance/test_pdc_x_requirements.py",
    "tests/unit/governance/test_pdc_x_wbs.py",
]
files += rel((root / "tests/unit/memory").rglob("test_*.py"))
files += rel((root / "tests/unit/agent_distribution").rglob("test_*.py"))
files += rel((root / "tests/qualification/compat_x").rglob("test_*.py"))
files += rel((root / "tests/unit/contracts").glob("*agent_contract*.py"))
seen, ordered = set(), []
for f in files:
    f = f.replace("\\", "/")
    if f not in seen and (root / f).is_file():
        seen.add(f)
        ordered.append(f)
print(f"files: {len(ordered)}", flush=True)

SUMMARY = re.compile(r"\b\d+ (passed|failed|skipped|xfailed|error)\b.* in [\d.]+s")
t0 = time.time()
proc = subprocess.Popen(
    [sys.executable, "-m", "pytest", *ordered, "-q", "-rfEx", "--tb=short"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=root,
)
state = {"summary_at": None}


def pump() -> None:
    for line in proc.stdout:  # type: ignore[union-attr]
        sys.stdout.write(line)
        sys.stdout.flush()
        if SUMMARY.search(line):
            state["summary_at"] = time.time()


th = threading.Thread(target=pump, daemon=True)
th.start()
status = None
while status is None:
    if proc.poll() is not None:
        status = "EXITED"
    elif state["summary_at"] and time.time() - state["summary_at"] > GRACE:
        status = "KILLED_AFTER_SUMMARY"
    elif time.time() - t0 > LIMIT:
        status = "TIMEOUT"
    else:
        time.sleep(1)
if status != "EXITED":
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
th.join(timeout=10)
print(f"RUNNER status={status} exit={proc.poll()} elapsed={time.time() - t0:.1f}s", flush=True)
