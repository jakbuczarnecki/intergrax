"""P7-00 mutation sample on highest-risk PDC-X paths (auditor-owned).

Usage: python p7_00_mutation_sample.py <scratch_checkout_root>

<scratch_checkout_root> must be a throw-away detached checkout of the audited SHA; the
auditor's own branch is never mutated. Each mutant is applied to one product file in the
scratch checkout, the named tests run in ONE pytest process (PYTHONPATH = scratch), the file
is restored with git checkout. A mutant is KILLED when pytest exits non-zero.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()

D6 = "tests/qualification/pdc_x/d6"
D5 = "tests/qualification/pdc_x/d5"
D8 = "tests/qualification/pdc_x/d8"
D1 = "tests/qualification/pdc_x/d1"

MUTANTS = [
    ("P1", "intergrax/applications/_shared/critical_action_signing_wiring.py",
     "if secret is None or not secret.strip():", "if secret is None:", [f"{D6}/test_d6_signing_fail_closed.py"]),
    ("P2", "intergrax/applications/_shared/critical_action_signing_wiring.py",
     "if lab_opt_in is not None and is_product:", "if False:", [f"{D6}/test_d6_signing_fail_closed.py"]),
    ("P3", "intergrax/core/plugins/admission.py",
     "if installed is None or installed != attestation.package_version:", "if False:", [f"{D6}/test_d6_plugin_provenance.py"]),
    ("P4", "intergrax/core/plugins/admission.py",
     "            if policy.forbid_unsigned_or_unqualified:", "            if False:", [f"{D6}/test_d6_plugin_provenance.py"]),
    ("P5", "intergrax/core/security/security_configuration.py",
     "return value is not False", "return value is True", [f"{D6}/test_d6_poisoning_default.py"]),
    ("P6", "intergrax/runtime/security/security_audit_trail.py",
     "if event.event_kind != SECURITY_AUDIT_EVENT_KIND or event.tenant_id != tenant:",
     "if event.event_kind != SECURITY_AUDIT_EVENT_KIND:", [f"{D6}/test_d6_audit_trail_durable.py"]),
    ("P7", "intergrax/applications/_shared/application_security_wiring.py",
     "                assert_tool_input_safe(value)", "                pass", [f"{D6}/test_d6_tool_input_defense_path.py"]),
    ("P8", "intergrax/applications/_shared/attestation_runtime_bridge.py",
     "lab_demo_seed_opt_in=not strict,", "lab_demo_seed_opt_in=True,", [f"{D6}/test_d6_demo_seed_gate.py"]),
    ("P9", "intergrax/runtime/nexus/artifacts/governed_access.py",
     "if found.ref.scope.tenant_id != context.tenant_id:", "if False:", [f"{D8}/test_d8_governed_port.py", f"{D8}/test_xd10_tenant_and_ttl.py"]),
    ("P10", "intergrax/runtime/nexus/artifacts/governed_access.py",
     "found.ref.verify_content(found.content)", "pass", [f"{D8}/test_d8_governed_port.py"]),
    ("P11", "intergrax/runtime/architecture/online_evaluation_registry.py",
     "return self._registry.list_observations_for_tenant(self._tenant_id)", "return self._registry.list_observations()",
     [f"{D5}/test_d5_evidence_and_tenant.py"]),
    ("P12", "intergrax/skills/host_lifecycle.py",
     "if prior_binding.package_reference != package_identity.package_reference:", "if False:", [f"{D1}/test_d1_skill_lifecycle.py"]),
]

WIDE = ["P6", "P9"]  # second pass: survivors of the targeted file re-run against the wider PDC-X suites
if len(sys.argv) > 2 and sys.argv[2] == "wide":
    wide_tests = [D6, "tests/qualification/pdc_x/p6_repair", "tests/qualification/pdc_x/xd", D8]
    MUTANTS = [(m, r, o, n, wide_tests) for (m, r, o, n, _t) in MUTANTS if m in WIDE]

results = []
for mid, rel, old, new, tests in MUTANTS:
    path = root / rel
    src = path.read_text(encoding="utf-8")
    count = src.count(old)
    if count != 1:
        results.append({"id": mid, "file": rel, "status": "NOT_APPLIED", "reason": f"pattern count {count}"})
        continue
    path.write_text(src.replace(old, new), encoding="utf-8")
    t = time.time()
    try:
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH", "")])))
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *tests, "-q", "-x", "--tb=line", "--ignore=tests/qualification/pdc_x/xd/test_xd_03_agent_governance.py", "--deselect=tests/qualification/pdc_x/d8/test_d8_collaborative_work_and_bundle.py::test_built_bundle_carries_digests_and_the_run_id"],
            cwd=root, env=env, capture_output=True, text=True, timeout=420,
        )
        tail = [ln for ln in proc.stdout.splitlines() if ln.strip()][-3:]
        results.append({"id": mid, "file": rel, "mutation": f"{old!r} -> {new!r}", "tests": tests,
                        "status": "KILLED" if proc.returncode != 0 else "SURVIVED",
                        "exit": proc.returncode, "seconds": round(time.time() - t, 1), "tail": tail})
    except subprocess.TimeoutExpired:
        results.append({"id": mid, "file": rel, "status": "TIMEOUT", "tests": tests})
    finally:
        subprocess.run(["git", "checkout", "--", rel], cwd=root, check=True)
    print(json.dumps(results[-1]), flush=True)

summary = {
    "killed": sum(r["status"] == "KILLED" for r in results),
    "survived": [r["id"] for r in results if r["status"] == "SURVIVED"],
    "not_applied": [r["id"] for r in results if r["status"] == "NOT_APPLIED"],
    "total": len(results),
}
print("SUMMARY " + json.dumps(summary))
