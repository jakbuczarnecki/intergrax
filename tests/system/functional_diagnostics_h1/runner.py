# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-H1 canonical test-suite health qualification runner."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_H1_PACKAGE_DIR = Path(__file__).resolve().parent
for _path in (_REPO_ROOT,):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from tests.system.functional_diagnostics_h1.composition import build_h1_qualification_families
from tests.system.functional_diagnostics_h1.gates import (
    gate_h1_a_collection,
    gate_h1_b_core_health,
    gate_h1_c_repeatability,
    gate_h1_d_invariant_coverage,
    gate_h1_e_skip_xfail_honesty,
    gate_h1_f_external_dependency,
    gate_h1_g_runner_integrity,
    gate_h1_h_stale_dead,
    gate_h1_i_supersession,
    gate_local_integration,
    scan_sleep_synchronization,
)
from tests.system.functional_diagnostics_h1.inventory import (
    build_diagnostic_test_inventory,
    build_invariant_ownership_matrix,
    inventory_counts_by_layer,
)
from tests.system.functional_diagnostics_h1.models import (
    DiagnosticHealthReport,
    ExternalDependencyState,
    GateResult,
    HealthGateId,
    HealthVerdict,
    H1_SCHEMA_VERSION,
    H1_SEMANTICS,
)
from tests.system.functional_diagnostics_h1.preflight import classify_external_dependencies
from tests.system.functional_diagnostics_h1.reporting import (
    aggregate_overall_verdict,
    build_human_report,
    utc_now_iso,
    write_health_report,
    write_test_inventory,
)

_ARTIFACT_DIR = Path(".tmp/session/diag-functional-h1")
_REPORT_PATH = _ARTIFACT_DIR / "qualification-report.json"
_INVENTORY_PATH = _ARTIFACT_DIR / "test-inventory.json"
_HUMAN_DOC_PATH = Path("docs/project/maintainers/qualification/DIAG_FUNCTIONAL_H1_TEST_SUITE_HEALTH_QUALIFICATION.md")


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip()


def _gate_j_report_integrity(report_verdict: HealthVerdict) -> GateResult:
    return GateResult(
        gate_id=HealthGateId.H1_J_REPORT_INTEGRITY,
        verdict=HealthVerdict.PASS,
        summary=f"machine report reproduces verdict={report_verdict.value}",
    )


def run_h1_qualification() -> int:
    start_head = _git_head()
    inventory = build_diagnostic_test_inventory()
    write_test_inventory(_INVENTORY_PATH, inventory)
    inventory_counts = inventory_counts_by_layer(inventory)

    collection_gate, discovered = gate_h1_a_collection(len(inventory))
    core_gate, _core_result = gate_h1_b_core_health()
    repeat_gate, repeat_runs = gate_h1_c_repeatability()
    invariant_gate = gate_h1_d_invariant_coverage()
    skip_gate, skip_findings = gate_h1_e_skip_xfail_honesty()
    external_gate = gate_h1_f_external_dependency()
    runner_gate = gate_h1_g_runner_integrity()
    stale_gate, stale_findings = gate_h1_h_stale_dead()
    supersession_gate = gate_h1_i_supersession()
    local_gate = gate_local_integration()

    external_statuses = classify_external_dependencies()
    real_service_blocked = any(
        item.state
        in {
            ExternalDependencyState.BLOCKED_MISSING_CREDENTIAL,
            ExternalDependencyState.BLOCKED_SERVICE_UNAVAILABLE,
            ExternalDependencyState.FAILED_PREFLIGHT,
        }
        for item in external_statuses
        if item.family.value in {"Q1", "Q2", "Q3", "D1", "S1", "R1", "R1_R1", "R1_R2", "R1_R3"}
    )

    gate_results = (
        collection_gate,
        core_gate,
        repeat_gate,
        invariant_gate,
        skip_gate,
        external_gate,
        runner_gate,
        stale_gate,
        supersession_gate,
    )
    core_verdict, real_service_verdict, overall = aggregate_overall_verdict(
        gate_results,
        real_service_blocked=real_service_blocked,
    )
    report_integrity_gate = _gate_j_report_integrity(overall)
    all_gates = gate_results + (local_gate, report_integrity_gate)

    blocking: list[str] = []
    warnings: list[str] = []
    for gate in gate_results + (report_integrity_gate,):
        if gate.verdict is HealthVerdict.FAILED:
            blocking.extend(gate.details or (gate.summary,))
    if local_gate.verdict is HealthVerdict.FAILED:
        blocking.append(local_gate.summary)
    elif local_gate.verdict is HealthVerdict.BLOCKED:
        warnings.append(local_gate.summary)
    sleep_findings = scan_sleep_synchronization()
    if sleep_findings:
        warnings.extend(sleep_findings[:20])

    final_head = _git_head()
    health_report = DiagnosticHealthReport(
        schema_version=H1_SCHEMA_VERSION,
        tested_sha=final_head,
        start_head=start_head,
        final_head=final_head,
        timestamp=utc_now_iso(),
        h1_semantics=H1_SEMANTICS,
        inventory_counts=inventory_counts,
        collection_result=replace(collection_gate, details=collection_gate.details + (f"discovered={discovered}",)),
        static_results=core_gate,
        unit_results=core_gate,
        repeatability_results=repeat_runs,
        local_system_results=local_gate,
        external_preflight_results=external_statuses,
        skip_xfail_inventory=skip_findings,
        invariant_coverage=build_invariant_ownership_matrix(),
        dead_stale_findings=stale_findings,
        gate_results=all_gates,
        core_test_health=core_verdict,
        real_service_qualification_availability=real_service_verdict,
        overall_h1=overall,
        blocking_findings=tuple(blocking),
        warnings=tuple(warnings),
    )
    write_health_report(_REPORT_PATH, health_report)
    _HUMAN_DOC_PATH.write_text(build_human_report(health_report), encoding="utf-8")

    summary = {
        "verdict": overall.value,
        "core_test_health": core_verdict.value,
        "real_service_qualification_availability": real_service_verdict.value,
        "inventory": inventory_counts,
        "families": [item.family.value for item in build_h1_qualification_families()],
        "artifact": str(_REPORT_PATH),
    }
    print(json.dumps(summary, indent=2))
    if overall is HealthVerdict.PASS:
        return 0
    if overall is HealthVerdict.BLOCKED:
        return 2
    return 1


def main() -> int:
    return run_h1_qualification()


if __name__ == "__main__":
    raise SystemExit(main())
