"""P7-00 addendum 3: ONE bounded pytest process for the regression run (auditor-owned).

Selection (list built from Python): MCP-RC mandatory contract, the new p6_ecapfix tests plus reg_p / p6_impact / xd_12,
the EBH-5 package C v2 file and the three EBH-5 gate files, the EBH-7 int2e bundle regression, the integration catalog /
registry / embedding-provider unit tests that import the changed module or its owning row, the pdc_x governance unit
tests (the two changed qualification scripts), the dependency foundation test (uv.lock changed) and the status files
of F-35 (COMPAT-X gates, agent distribution) and F-07. xd_03 (F-09, exit hang) is run in its own process afterwards.
The process is killed GRACE seconds after pytest printed its final summary line (F-09 guard).
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
from tests.qualification.mcp_server_x.rc_selection import MCP_SERVER_X_RC_SELECTION  # noqa: E402

GRACE = 30
LIMIT = 4500
U = "tests/unit/integrations"
files = list(MCP_SERVER_X_RC_SELECTION)
files += [
    "tests/qualification/pdc_x/p6_ecapfix",
    "tests/qualification/pdc_x/reg_p",
    "tests/qualification/pdc_x/p6_impact",
    "tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py",
    "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py",
    "tests/qualification/ebh_5/test_ebh_5_g06_enterprise_integrity_gates.py",
    "tests/qualification/ebh_5/test_ebh_5_proof_integrity_gates.py",
    "tests/qualification/ebh_5/test_ebh_5_rpl_f_integrity_gates.py",
    "tests/qualification/ebh_7/test_ebh_7_int2e_bundle_regression.py",
    f"{U}/registry",
    f"{U}/contracts",
    f"{U}/providers/managed_retrieval",
    f"{U}/providers/test_p5_m6_p4_providers.py",
    f"{U}/providers/test_interaction_surface_taxonomy_removal.py",
    f"{U}/providers/test_provider_category_contract_migration.py",
    f"{U}/providers/test_provider_runtime_cutover.py",
    f"{U}/providers/test_provider_scaffold_contract_pattern.py",
    f"{U}/test_p2_002_b1_embedding_provider_category.py",
    f"{U}/test_p2_002_b2_embedding_provider_catalog.py",
    f"{U}/test_p2_002_b3_runtime_binding_extension.py",
    f"{U}/test_p2_003_c_final_cutover.py",
    f"{U}/test_p2_003_explicit_contract_specs.py",
    f"{U}/test_registry.py",
    f"{U}/test_profile.py",
    f"{U}/test_profile_selected_provider_materialization.py",
    f"{U}/test_resolve_typed.py",
    f"{U}/test_integration_presets.py",
    f"{U}/test_vendor_import_governance.py",
    f"{U}/test_contract_conformance.py",
    f"{U}/test_bootstrap_h_int_graph.py",
    f"{U}/test_harness_lab_health.py",
    "tests/unit/rag/bootstrap",
    "tests/unit/rag/document_splitters/test_optional_langchain_splitter.py",
    "tests/unit/test_dep1_dependency_foundation.py",
    "tests/unit/governance/test_pdc_x_inventory.py",
    "tests/unit/governance/test_pdc_x_requirements.py",
    "tests/unit/governance/test_pdc_x_wbs.py",
    # status files of F-35 (COMPAT-X gates and agent distribution, red since PDC-X in addendum 1)
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
]
seen: set[str] = set()
ordered: list[str] = []
missing: list[str] = []
for f in files:
    f = f.replace("\\", "/")
    if f in seen:
        continue
    seen.add(f)
    (ordered if (root / f).exists() else missing).append(f)
print(f"entries: {len(ordered)} missing: {missing}", flush=True)

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
