"""P7-00 addendum 2: ONE bounded pytest process for the regression run (auditor-owned).

Contents: MCP-RC mandatory contract (list from Python), the tests that touch the CHANGED-SINCE impact map
(tests/qualification/ent_audit_x, evol_x, roadmap_replay_x, pdc_x/p6_impact, pdc_x xd_12, test_pdc_x_*), the EBH-5 gate files
(F-06 follow-up) and the pytest entrypoints of every manifest check that the new PDC domain rules select (so that a
CHANGED-SINCE run triggered by the new rules is itself runnable). The process is killed GRACE seconds after pytest printed
its final summary line (known pytest exit hang, finding F-09 of the first audit).
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root))
from intergrax.platform.enterprise_audit.impact_map import PDC_DOMAIN_IMPACT_RULES  # noqa: E402
from intergrax.platform.enterprise_audit.manifest import load_default_manifest  # noqa: E402
from tests.qualification.mcp_server_x.rc_selection import MCP_SERVER_X_RC_SELECTION  # noqa: E402

GRACE = 30
LIMIT = 3400


def rel(paths):
    return sorted({str(p.relative_to(root)).replace("\\", "/") for p in paths})


files = list(MCP_SERVER_X_RC_SELECTION)
files += rel((root / "tests/qualification/ent_audit_x").rglob("test_*.py"))
files += rel((root / "tests/qualification/evol_x").rglob("test_*.py"))
files += rel((root / "tests/qualification/roadmap_replay_x").rglob("test_*.py"))
files += rel((root / "tests/qualification/pdc_x/p6_impact").rglob("test_*.py"))
files += [
    "tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py",
    "tests/qualification/ebh_5/test_ebh_5_g06_enterprise_integrity_gates.py",
    "tests/qualification/ebh_5/test_ebh_5_proof_integrity_gates.py",
    "tests/qualification/ebh_5/test_ebh_5_rpl_f_integrity_gates.py",
    "tests/unit/governance/test_pdc_x_inventory.py",
    "tests/unit/governance/test_pdc_x_requirements.py",
    "tests/unit/governance/test_pdc_x_wbs.py",
]
# entrypoints of the manifest checks selected by the new domain rules (pytest ones only; scripts are run separately)
rule_ids = {i for _, ids in PDC_DOMAIN_IMPACT_RULES for i in ids}
for check in load_default_manifest():
    if check.check_id in rule_ids and check.entrypoint.replace("\\", "/").split("::", 1)[0].startswith("tests/"):
        files.append(check.entrypoint.replace("\\", "/"))

seen, ordered, files_only = set(), [], set()
for f in files:
    f = f.replace("\\", "/")
    if f in seen or not (root / f.split("::", 1)[0]).is_file():
        continue
    seen.add(f)
    # a whole file already present makes its node-id entries redundant
    ordered.append(f)
    if "::" not in f:
        files_only.add(f)
ordered = [f for f in ordered if "::" not in f or f.split("::", 1)[0] not in files_only]
print(f"entries: {len(ordered)}", flush=True)

SUMMARY = re.compile(r"\b\d+ (passed|failed|skipped|xfailed|error)\b.* in [\d.]+s")
t0 = time.time()
proc = subprocess.Popen(
    [sys.executable, "-m", "pytest", *ordered, "-q", "-rfEx", "--tb=short"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=root,
)
state: dict[str, float | None] = {"summary_at": None}


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
