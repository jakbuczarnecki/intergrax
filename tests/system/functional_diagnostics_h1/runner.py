# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-H1 canonical test-suite health qualification runner."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
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
    gate_h1_k_local_integration,
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
    H1_QUALIFICATION_ID,
    H1_SCHEMA_VERSION,
    H1_SEMANTICS,
    QualificationRepositoryTransition,
)
from tests.system.functional_diagnostics_h1.preflight import classify_external_dependencies
from tests.system.functional_diagnostics_h1.qualification_spec import (
    DiagnosticHealthQualificationSpec,
    resolve_qualification_spec,
)
from tests.system.functional_diagnostics_h1.reporting import (
    aggregate_overall_verdict,
    build_human_report,
    gate_h1_j_report_integrity,
    utc_now_iso,
    write_health_report,
    write_test_inventory,
)
from tests.system.functional_diagnostics_h1.repository_state import (
    assert_qualification_repository_postconditions,
    assert_qualification_repository_state,
    capture_qualification_repository_state,
)

_EXIT_PASS = 0
_EXIT_FAILED = 1
_EXIT_BLOCKED = 2
_EXIT_FAILED_PRECONDITION = 3


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip()


def _collect_blocking_findings(gate_results: tuple[GateResult, ...]) -> tuple[str, ...]:
    blocking: list[str] = []
    for gate in gate_results:
        if gate.verdict is HealthVerdict.FAILED:
            blocking.extend(gate.details or (gate.summary,))
    return tuple(blocking)


def _write_artifact_human_report(
    spec: DiagnosticHealthQualificationSpec,
    report: DiagnosticHealthReport,
) -> None:
    human_path = spec.artifact_human_report_md
    human_path.parent.mkdir(parents=True, exist_ok=True)
    human_path.write_text(build_human_report(report, spec), encoding="utf-8")


def _write_precondition_failure_report(
    *,
    spec: DiagnosticHealthQualificationSpec,
    repo_transition: QualificationRepositoryTransition,
    precondition: HealthVerdict,
    violations: tuple[str, ...],
) -> DiagnosticHealthReport:
    placeholder = GateResult(
        gate_id=HealthGateId.H1_J_REPORT_INTEGRITY,
        verdict=HealthVerdict.FAILED,
        summary="qualification aborted: repository precondition failed",
        details=violations,
    )
    start_state = repo_transition.start
    health_report = DiagnosticHealthReport(
        schema_version=H1_SCHEMA_VERSION,
        qualification_id=spec.qualification_id,
        tested_sha=start_state.head_sha,
        start_head=start_state.head_sha,
        final_head=repo_transition.end.head_sha,
        origin_development_sha=start_state.origin_development_sha,
        origin_development_at_end=repo_transition.end.origin_development_sha,
        working_tree_clean_at_start=start_state.working_tree_clean,
        working_tree_clean_at_end=repo_transition.end.working_tree_clean,
        repository_precondition=precondition,
        repository_postcondition=HealthVerdict.FAILED,
        timestamp=utc_now_iso(),
        h1_semantics=H1_SEMANTICS,
        inventory_counts={},
        collection_result=placeholder,
        static_results=placeholder,
        unit_results=placeholder,
        repeatability_results=(),
        local_system_results=placeholder,
        external_preflight_results=(),
        skip_xfail_inventory=(),
        invariant_coverage=(),
        dead_stale_findings=(),
        gate_results=(placeholder,),
        core_test_health=HealthVerdict.FAILED,
        real_service_qualification_availability=HealthVerdict.BLOCKED,
        overall_h1=HealthVerdict.FAILED_PRECONDITION,
        blocking_findings=violations,
        warnings=(),
    )
    write_health_report(spec.artifact_report_json, health_report)
    write_test_inventory(spec.artifact_inventory_json, ())
    _write_artifact_human_report(spec, health_report)
    return health_report


def run_h1_qualification(
    *,
    qualification_id: str = H1_QUALIFICATION_ID,
    artifact_dir: Path | None = None,
    human_doc_path: Path | None = None,
    skip_repository_preconditions: bool = False,
) -> int:
    spec = resolve_qualification_spec(qualification_id)
    if artifact_dir is not None:
        spec = DiagnosticHealthQualificationSpec(
            qualification_id=spec.qualification_id,
            artifact_directory=artifact_dir,
            closure_doc_path=spec.closure_doc_path,
            requires_clean_repository=spec.requires_clean_repository,
            requires_origin_development_match=spec.requires_origin_development_match,
            requires_stable_head=spec.requires_stable_head,
            historical=spec.historical,
            requires_closure_doc_at_run=spec.requires_closure_doc_at_run,
        )
    if human_doc_path is not None:
        _ = human_doc_path

    start_state = capture_qualification_repository_state(_REPO_ROOT)
    precondition, precondition_violations = assert_qualification_repository_state(
        start_state,
        requires_clean_repository=spec.requires_clean_repository,
        requires_origin_development_match=spec.requires_origin_development_match,
    )
    if (
        not skip_repository_preconditions
        and spec.requires_repository_preconditions()
        and precondition is not HealthVerdict.PASS
    ):
        end_state = capture_qualification_repository_state(_REPO_ROOT)
        _write_precondition_failure_report(
            spec=spec,
            repo_transition=QualificationRepositoryTransition(start=start_state, end=end_state),
            precondition=precondition,
            violations=precondition_violations,
        )
        summary = {
            "qualification_id": spec.qualification_id,
            "verdict": HealthVerdict.FAILED_PRECONDITION.value,
            "violations": precondition_violations,
            "artifact": str(spec.artifact_report_json),
        }
        print(json.dumps(summary, indent=2))
        return _EXIT_FAILED_PRECONDITION

    start_head = start_state.head_sha
    inventory = build_diagnostic_test_inventory()
    write_test_inventory(spec.artifact_inventory_json, inventory)
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
    local_gate = gate_h1_k_local_integration()

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

    gate_results_before_j = (
        collection_gate,
        core_gate,
        repeat_gate,
        invariant_gate,
        skip_gate,
        external_gate,
        runner_gate,
        stale_gate,
        supersession_gate,
        local_gate,
    )
    core_verdict, real_service_verdict, overall = aggregate_overall_verdict(
        gate_results_before_j,
        real_service_blocked=real_service_blocked,
    )
    blocking = _collect_blocking_findings(gate_results_before_j)

    end_state = capture_qualification_repository_state(_REPO_ROOT)
    final_head = end_state.head_sha
    repo_transition = QualificationRepositoryTransition(start=start_state, end=end_state)
    postcondition, postcondition_violations = assert_qualification_repository_postconditions(
        repo_transition,
        spec,
    )
    if postcondition is not HealthVerdict.PASS:
        blocking = blocking + postcondition_violations
        overall = HealthVerdict.FAILED
        core_verdict = HealthVerdict.FAILED

    report_integrity_gate = gate_h1_j_report_integrity(
        gate_results=gate_results_before_j,
        calculated_overall=overall,
        blocking_findings=blocking,
        start_head=start_head,
        final_head=final_head,
        working_tree_clean_at_end=end_state.working_tree_clean,
        repository_postcondition=postcondition,
    )
    all_gates = gate_results_before_j + (report_integrity_gate,)

    if report_integrity_gate.verdict is HealthVerdict.FAILED:
        overall = HealthVerdict.FAILED
        core_verdict = HealthVerdict.FAILED
        blocking = _collect_blocking_findings(all_gates) + postcondition_violations

    warnings: list[str] = []
    sleep_findings = scan_sleep_synchronization()
    if sleep_findings:
        warnings.extend(sleep_findings[:20])

    health_report = DiagnosticHealthReport(
        schema_version=H1_SCHEMA_VERSION,
        qualification_id=spec.qualification_id,
        tested_sha=start_head,
        start_head=start_head,
        final_head=final_head,
        origin_development_sha=start_state.origin_development_sha,
        origin_development_at_end=end_state.origin_development_sha,
        working_tree_clean_at_start=start_state.working_tree_clean,
        working_tree_clean_at_end=end_state.working_tree_clean,
        repository_precondition=precondition,
        repository_postcondition=postcondition,
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
        blocking_findings=blocking,
        warnings=tuple(warnings),
    )
    write_health_report(spec.artifact_report_json, health_report)
    _write_artifact_human_report(spec, health_report)

    summary = {
        "qualification_id": spec.qualification_id,
        "verdict": overall.value,
        "core_test_health": core_verdict.value,
        "real_service_qualification_availability": real_service_verdict.value,
        "tested_sha": start_head,
        "start_head": start_head,
        "final_head": final_head,
        "origin_development_at_start": start_state.origin_development_sha,
        "origin_development_at_end": end_state.origin_development_sha,
        "working_tree_clean_at_start": start_state.working_tree_clean,
        "working_tree_clean_at_end": end_state.working_tree_clean,
        "repository_precondition": precondition.value,
        "repository_postcondition": postcondition.value,
        "inventory": inventory_counts,
        "families": [item.family.value for item in build_h1_qualification_families()],
        "artifact": str(spec.artifact_report_json),
    }
    print(json.dumps(summary, indent=2))
    if overall is HealthVerdict.PASS:
        return _EXIT_PASS
    if overall is HealthVerdict.BLOCKED:
        return _EXIT_BLOCKED
    if overall is HealthVerdict.FAILED_PRECONDITION:
        return _EXIT_FAILED_PRECONDITION
    return _EXIT_FAILED


def main() -> int:
    qualification_id = H1_QUALIFICATION_ID
    skip_repository_preconditions = False
    args = sys.argv[1:]
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--qualification-id" and index + 1 < len(args):
            qualification_id = args[index + 1]
            index += 2
            continue
        if arg == "--skip-repository-preconditions":
            skip_repository_preconditions = True
            index += 1
            continue
        index += 1
    return run_h1_qualification(
        qualification_id=qualification_id,
        skip_repository_preconditions=skip_repository_preconditions,
    )


if __name__ == "__main__":
    raise SystemExit(main())
