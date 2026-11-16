"""IA-D1 mutation sample: applies one mutant at a time to a THROWAWAY checkout (never the audited worktree),
runs the D1 pdc_x test files once per mutant (one pytest process at a time), restores the file.

Usage: python run_mutants.py <mutation_worktree> <python_exe> <log_dir>
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

WT = Path(sys.argv[1])
PY = sys.argv[2]
LOG = Path(sys.argv[3])
LOG.mkdir(parents=True, exist_ok=True)
ADMIN = WT / "intergrax/agent_distribution/admin_service.py"
ADM = WT / "intergrax/agent_distribution/upgrade_admission.py"
INSTSVC = WT / "intergrax/agent_distribution/installation_service.py"

MUTANTS = [
    ("M1-admission-bypass", ADMIN, "        if not report.admitted:\n", "        if False:\n"),
    (
        "M2-no-policy-authz-upgrade-retire",
        ADMIN,
        "        resolver = self._require_environment_tenant_resolver()\n        boundary = self._require_mutation_authorization_boundary()\n        return self._enforce_authorization_result(\n            authorize_scoped_control_plane_mutation(",
        "        if operation in {'upgrade_agent', 'retire_agent'}:\n            return None  # MUTANT\n        resolver = self._require_environment_tenant_resolver()\n        boundary = self._require_mutation_authorization_boundary()\n        return self._enforce_authorization_result(\n            authorize_scoped_control_plane_mutation(",
    ),
    (
        "M3-retire-no-app-scope",
        ADMIN,
        "        self._assert_installation_application_scope(\n            application_id=application_id,\n            application_environment_id=application_environment_id,\n            record=active,\n        )\n",
        "",
    ),
    (
        "M4-retire-no-tenant-scope",
        ADMIN,
        "        resolver = self._require_environment_tenant_resolver()\n        environment_tenant = resolver.resolve_tenant_id(",
        "        if operation == 'retire_agent':\n            return  # MUTANT\n        resolver = self._require_environment_tenant_resolver()\n        environment_tenant = resolver.resolve_tenant_id(",
    ),
    (
        "M5-upgrade-no-tenant-scope",
        ADMIN,
        "        resolver = self._require_environment_tenant_resolver()\n        environment_tenant = resolver.resolve_tenant_id(",
        "        if operation == 'upgrade_agent':\n            return  # MUTANT\n        resolver = self._require_environment_tenant_resolver()\n        environment_tenant = resolver.resolve_tenant_id(",
    ),
    (
        "M6-retired-slot-accepts-candidates",
        INSTSVC,
        "        self._require_slot_not_retired(environment_id, installation_slot_id)\n        record = AgentInstallationRecord(",
        "        record = AgentInstallationRecord(",
    ),
    ("M7-same-version-is-upgrade", ADM, "    if candidate_key <= prior_key:", "    if candidate_key < prior_key:"),
    (
        "M8-bound-tool-removal-ignored",
        ADM,
        "            if tool_id in prior_tools and tool_id not in candidate_tools:",
        "            if False:",
    ),
    (
        "M9-capability-removal-ignored",
        ADM,
        "        if removed_capabilities:\n",
        "        if False:\n",
    ),
    (
        "M10-failed-candidate-not-recorded-on-reject",
        ADMIN,
        "            failed = self._installation_service.mark_failed_candidate(\n                request.installation_id,\n                reason_code=reason.value if reason is not None else None,\n            )\n",
        "            failed = self._installation_service.mark_failed_candidate(\n                request.installation_id,\n                reason_code=None,\n            )\n",
    ),
]

TESTS = [
    "tests/qualification/pdc_x/d1/test_d1_adversarial_lifecycle.py",
    "tests/qualification/pdc_x/d1/test_d1_upgrade_retire.py",
]

summary: list[str] = []
for name, path, old, new in MUTANTS:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count < 1:
        summary.append(f"{name}: PATTERN NOT FOUND")
        continue
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [PY, "-m", "pytest", *TESTS, "-q", "-x", "--tb=line", f"--basetemp={LOG}/bt-{name}"],
            cwd=WT,
            capture_output=True,
            text=True,
            timeout=500,
        )
        tail = [line for line in proc.stdout.splitlines() if line.strip()][-3:]
        verdict = "KILLED" if proc.returncode != 0 else "SURVIVED"
        (LOG / f"{name}.log").write_text(proc.stdout[-6000:], encoding="utf-8")
        summary.append(f"{name}: {verdict} (rc={proc.returncode}, {time.time() - t0:.0f}s) :: {tail[-1] if tail else ''}")
    except subprocess.TimeoutExpired:
        summary.append(f"{name}: TIMEOUT")
    finally:
        path.write_text(text, encoding="utf-8")
(LOG / "mutation_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
print("\n".join(summary))
