# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q2 real tool-selection qualification orchestrator."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from intergrax.contracts.execution_identity import validate_run_id, validate_task_id
from intergrax.tools.providers.workspace.service import WORKSPACE_SEARCH_TOOL_ID
from intergrax.core.qualification.functional_diagnostic_comparator import compare_qualification_case
from intergrax.core.qualification.functional_diagnostic_expectation import (
    QualificationCaseComparison,
    QualificationCaseExpectation,
    QualificationComparisonResult,
    QualificationExecutionOutcome,
    QualificationFunctionalOutcome,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessment
from intergrax.runtime.diagnostics.diagnostic_assessment_composer import DiagnosticAssessmentComposer
from intergrax.runtime.diagnostics.functional_diagnostic_analyzer import FunctionalDiagnosticAnalyzer
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import (
    FunctionalDiagnosticCheckStatus,
)
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
    PipelineEvidenceScope,
    PipelineOperationStatus,
    PlatformFunctionalEvidence,
)
from intergrax.runtime.diagnostics.functional_validation_lookup import FunctionalValidationEvidenceLookup
from intergrax.runtime.diagnostics.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from intergrax.runtime.diagnostics.specifications.q2_tool_selection_functional_diagnostic_specification import (
    Q2_EXPECTED_SEARCH_TOOL_ARTIFACT,
    Q2_TOOL_INVOKE_OPERATION_ID,
    build_q2_tool_selection_functional_diagnostic_specification,
)
from intergrax.runtime.observability.functional_evidence_runtime_wiring import (
    wire_in_memory_functional_evidence_runtime,
)
from tests.system.functional_diagnostics_q2.cases import (
    HEALTHY_TASK,
    MANDATORY_CASES,
    Q2_A_HEALTHY,
    Q2_B_WRONG_TOOL,
    Q2_F_HEALTHY,
    Q2_F_WRONG_TOOL,
    _REPEAT_CASE_ID,
    case_metadata,
)
from tests.system.functional_diagnostics_q2.oracle import (
    EXPECTED_SEARCH_TOOL_ARTIFACT,
    build_independent_validation_evidence,
    independent_tool_oracle_passes,
)
from tests.system.unified_execution.proof_runner.contracts import ProofConfig
from tests.system.unified_execution.proof_runner.lkw_client import LkwClient, LkwClientError, LkwRunResponse

_ARTIFACT_DIR = Path(
    os.environ.get(
        "DIAG_FUNCTIONAL_Q2_ARTIFACT_DIR",
        ".tmp/session/diag-functional-q2",
    ),
)
_CURSOR_SECRET = "diag-functional-q2-local-only-secret-32bytes!!"


@dataclass(frozen=True, slots=True)
class EvidenceFidelitySnapshot:
    provider_candidate_refs: tuple[str, ...]
    actual_selected_tool: str | None
    emitted_selected_tool: str | None
    actual_invoke_succeeded: bool | None
    emitted_invoke_succeeded: bool | None
    candidate_fidelity_match: bool
    selection_fidelity_match: bool
    invocation_fidelity_match: bool
    validation_fidelity_match: bool
    identity_fidelity_match: bool
    failure_injection_layer: str | None


@dataclass(frozen=True, slots=True)
class QualificationRunRecord:
    case_id: str
    task_id: str
    run_id: str
    execution_outcome: QualificationExecutionOutcome
    functional_outcome: QualificationFunctionalOutcome
    comparison: QualificationCaseComparison
    evidence_fidelity: EvidenceFidelitySnapshot
    diag_first_failed_check: str | None
    operator_outcome: str | None
    available_tools: tuple[str, ...]
    expected_tool: str
    actual_tool: str | None
    invocation_status: str | None
    repeat_group: str | None = None


@dataclass(frozen=True, slots=True)
class QualificationReport:
    verdict: str
    total_cases: int
    matched_cases: int
    mismatched_cases: int
    false_positive_cases: int
    false_negative_cases: int
    inconclusive_correct_cases: int
    stage_accuracy_percent: float
    inconclusive_accuracy_percent: float
    repeatability_pass: bool
    records: tuple[QualificationRunRecord, ...]
    blocked_reason: str | None = None


def _config_from_env() -> ProofConfig:
    return ProofConfig(
        base_url=os.environ.get("LKW_BASE_URL", "http://localhost:8021"),
        api_key=os.environ.get(
            "LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY",
            "ue-11g-c1-certification-secret",
        ),
        tenant_id=os.environ.get("DIAG_FUNCTIONAL_Q2_TENANT_ID", "tenant-ue-11g-c1"),
        workspace_id=os.environ.get("DIAG_FUNCTIONAL_Q2_WORKSPACE_ID", "ue-11g-c1-workspace"),
        collection_id=os.environ.get("DIAG_FUNCTIONAL_Q2_COLLECTION_ID", "ue-11g-c1-collection"),
        fixture_root=os.environ.get("DIAG_FUNCTIONAL_Q2_FIXTURE_ROOT", "/cert-fixtures/workspace"),
    )


def _fetch_functional_evidence(
    config: ProofConfig,
    *,
    tenant_id: str,
    task_id: str,
    run_id: str,
) -> tuple[PlatformFunctionalEvidence, ...]:
    query = urllib.parse.urlencode(
        {"tenant_id": tenant_id, "task_id": task_id, "run_id": run_id},
    )
    url = f"{config.base_url.rstrip('/')}/v1/local_workspace/qualification/functional_evidence?{query}"
    request = urllib.request.Request(url, headers={"X-API-Key": config.api_key})
    with urllib.request.urlopen(request, timeout=120.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    items = payload.get("items")
    if not isinstance(items, list):
        raise LkwClientError("functional_evidence_items_missing")
    return tuple(
        PlatformFunctionalEvidence.model_validate(_normalize_remote_evidence_item(item))
        for item in items
    )


def _normalize_remote_evidence_item(item: object) -> object:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    provenance = normalized.get("provenance")
    if isinstance(provenance, dict):
        prov = dict(provenance)
        upstream = prov.get("upstream_evidence_ids")
        if isinstance(upstream, list):
            prov["upstream_evidence_ids"] = tuple(upstream)
        normalized["provenance"] = prov
    return normalized


def _candidate_refs_from_items(items: tuple[PlatformFunctionalEvidence, ...]) -> tuple[str, ...]:
    refs: list[str] = []
    for item in items:
        if item.kind is not PipelineEvidenceKind.CANDIDATE_RANK or item.candidate is None:
            continue
        refs.append(item.candidate.candidate_artifact_ref.artifact_ref)
    return tuple(refs)


def _selection_ref_from_items(items: tuple[PlatformFunctionalEvidence, ...]) -> str | None:
    for item in items:
        if item.kind is PipelineEvidenceKind.SELECTION and item.selection is not None:
            return item.selection.selected_artifact_ref.artifact_ref
    return None


def _invoke_succeeded_from_items(items: tuple[PlatformFunctionalEvidence, ...]) -> bool | None:
    for item in items:
        if item.kind is not PipelineEvidenceKind.OPERATION_OUTCOME or item.operation_outcome is None:
            continue
        if item.provenance.operation_id != Q2_TOOL_INVOKE_OPERATION_ID:
            continue
        return item.operation_outcome.status is PipelineOperationStatus.SUCCEEDED
    return None


def _actual_tool_from_response(response: LkwRunResponse) -> str | None:
    if response.lkw_evidence is None:
        return None
    diagnostics = response.lkw_evidence.diagnostics
    summary = diagnostics.get("tool_selection_summary")
    if not isinstance(summary, dict):
        return None
    selected = summary.get("selected_tool_id")
    return str(selected).strip() if isinstance(selected, str) and selected.strip() else None


def _invoke_status_from_response(response: LkwRunResponse) -> str | None:
    if response.lkw_evidence is None:
        return None
    diagnostics = response.lkw_evidence.diagnostics
    summary = diagnostics.get("tool_selection_summary")
    if not isinstance(summary, dict):
        return None
    status = summary.get("invoke_status")
    return str(status) if status is not None else None


def _available_tools_from_response(response: LkwRunResponse) -> tuple[str, ...]:
    if response.lkw_evidence is None:
        return ()
    diagnostics = response.lkw_evidence.diagnostics
    summary = diagnostics.get("tool_selection_summary")
    if not isinstance(summary, dict):
        return ()
    raw = summary.get("available_tool_ids")
    if not isinstance(raw, list):
        return ()
    return tuple(str(item) for item in raw if isinstance(item, str))


def _expected_tool_for_case(case_id: str) -> str:
    _ = case_id
    return EXPECTED_SEARCH_TOOL_ARTIFACT


def _evidence_fidelity_snapshot(
    *,
    provider_candidates: tuple[str, ...],
    actual_selected: str | None,
    emitted_selected: str | None,
    actual_invoke_succeeded: bool | None,
    emitted_invoke_succeeded: bool | None,
    remote_items: tuple[PlatformFunctionalEvidence, ...],
    scope: PipelineEvidenceScope,
    failure_injection_layer: str | None,
    validation_expected: bool,
    validation_actual_pass: bool,
) -> EvidenceFidelitySnapshot:
    emitted_candidates = _candidate_refs_from_items(remote_items)
    identity_ok = all(
        item.scope.tenant_id == scope.tenant_id
        and item.scope.task_id == scope.task_id
        and item.scope.run_id == scope.run_id
        for item in remote_items
    )
    return EvidenceFidelitySnapshot(
        provider_candidate_refs=provider_candidates,
        actual_selected_tool=actual_selected,
        emitted_selected_tool=emitted_selected,
        actual_invoke_succeeded=actual_invoke_succeeded,
        emitted_invoke_succeeded=emitted_invoke_succeeded,
        candidate_fidelity_match=provider_candidates == emitted_candidates,
        selection_fidelity_match=actual_selected == emitted_selected,
        invocation_fidelity_match=actual_invoke_succeeded == emitted_invoke_succeeded,
        validation_fidelity_match=(
            validation_expected == validation_actual_pass if validation_expected else True
        ),
        identity_fidelity_match=identity_ok,
        failure_injection_layer=failure_injection_layer,
    )


def _semantic_signature(record: QualificationRunRecord) -> tuple[str, ...]:
    failed_checks = tuple(
        f"{item.check_id}:{item.expected_status.value}->{item.actual_status.value}"
        for item in record.comparison.check_mismatches
    )
    return (
        record.comparison.result.value,
        record.actual_tool or "",
        record.invocation_status or "",
        record.diag_first_failed_check or "",
        record.operator_outcome or "",
        ",".join(failed_checks),
    )


def _run_case(
    client: LkwClient,
    config: ProofConfig,
    expectation: QualificationCaseExpectation,
    *,
    repeat_group: str | None = None,
) -> QualificationRunRecord:
    metadata = {
        "tenant_id": config.tenant_id,
        "workspace_id": config.workspace_id,
        **case_metadata(expectation),
    }
    failure_layer_raw = metadata.get("qualification_failure_injection_layer")
    failure_layer = str(failure_layer_raw) if failure_layer_raw is not None else None

    response = client.run_tool_selection_qualification(
        message=str(metadata.get("qualification_task_message") or HEALTHY_TASK),
        metadata=metadata,
    )

    execution_outcome = (
        QualificationExecutionOutcome.COMPLETED
        if response.state == "completed"
        else QualificationExecutionOutcome.FAILED
    )
    remote_items = _fetch_functional_evidence(
        config,
        tenant_id=config.tenant_id,
        task_id=response.task_id,
        run_id=response.run_id,
    )
    provider_candidates = _candidate_refs_from_items(remote_items)
    actual_tool_ref = _selection_ref_from_items(remote_items)
    emitted_selected = actual_tool_ref
    actual_invoke = _invoke_succeeded_from_items(remote_items)
    emitted_invoke = actual_invoke

    functional_outcome = (
        QualificationFunctionalOutcome.PASSED
        if independent_tool_oracle_passes(
            answer=response.answer,
            selected_tool_artifact=actual_tool_ref,
        )
        else QualificationFunctionalOutcome.FAILED
    )

    pipeline_scope = PipelineEvidenceScope(
        tenant_id=config.tenant_id,
        task_id=validate_task_id(response.task_id),
        run_id=validate_run_id(response.run_id),
    )
    validation = build_independent_validation_evidence(
        pipeline_scope,
        answer=response.answer,
        selected_tool_artifact=actual_tool_ref,
        idempotency_key=expectation.case_id,
    )
    spec = build_q2_tool_selection_functional_diagnostic_specification(
        validation_id=validation.validation_id if expectation.include_validation else None,
        include_validation=expectation.include_validation,
        expected_selection_artifact_ref=Q2_EXPECTED_SEARCH_TOOL_ARTIFACT,
    )
    wiring = wire_in_memory_functional_evidence_runtime(cursor_secret=_CURSOR_SECRET)
    for item in remote_items:
        wiring.persistence.append(item)
    validations_lookup = (
        FunctionalValidationEvidenceLookup.for_scope(
            tenant_id=config.tenant_id,
            task_id=pipeline_scope.task_id,
            run_id=pipeline_scope.run_id,
            attempt_id=None,
            validations=(validation,),
        )
        if expectation.include_validation
        else None
    )
    analysis = FunctionalDiagnosticAnalyzer(wiring.persistence).analyze(
        tenant_id=config.tenant_id,
        task_id=pipeline_scope.task_id,
        run_id=pipeline_scope.run_id,
        specification=spec,
        validations=validations_lookup,
    )
    lifecycle = DiagnosticAssessment(
        tenant_id=config.tenant_id,
        task_id=pipeline_scope.task_id,
        run_id=pipeline_scope.run_id,
        findings=(),
        limitations=(),
    )
    operator = DiagnosticAssessmentComposer().compose(
        lifecycle_assessment=lifecycle,
        functional_analysis=analysis,
    )
    comparison = compare_qualification_case(
        expectation,
        actual_execution_outcome=execution_outcome,
        actual_functional_outcome=functional_outcome,
        analysis=analysis,
        operator_assessment=operator,
    )
    fidelity = _evidence_fidelity_snapshot(
        provider_candidates=provider_candidates,
        actual_selected=actual_tool_ref,
        emitted_selected=emitted_selected,
        actual_invoke_succeeded=actual_invoke,
        emitted_invoke_succeeded=emitted_invoke,
        remote_items=remote_items,
        scope=pipeline_scope,
        failure_injection_layer=failure_layer,
        validation_expected=expectation.include_validation,
        validation_actual_pass=validation.outcome.value == "passed",
    )
    return QualificationRunRecord(
        case_id=expectation.case_id,
        task_id=response.task_id,
        run_id=response.run_id,
        execution_outcome=execution_outcome,
        functional_outcome=functional_outcome,
        comparison=comparison,
        evidence_fidelity=fidelity,
        diag_first_failed_check=(
            str(analysis.first_proven_failure) if analysis.first_proven_failure is not None else None
        ),
        operator_outcome=(
            operator.functional_projection.outcome_status.value
            if operator.functional_projection is not None
            else None
        ),
        available_tools=provider_candidates,
        expected_tool=_expected_tool_for_case(expectation.case_id),
        actual_tool=actual_tool_ref,
        invocation_status=_invoke_status_from_response(response),
        repeat_group=repeat_group,
    )


def _decision_diagnostics_independence_gate() -> bool:
    import ast

    selection_path = (
        Path(__file__).resolve().parents[3]
        / "agents"
        / "tool_selection_qualifier"
        / "steps"
        / "tool_selection_job.py"
    )
    tree = ast.parse(selection_path.read_text(encoding="utf-8"))
    forbidden = (
        "intergrax.runtime.diagnostics",
        "functional_diagnostic",
        "functional_diagnostics_q2",
        "qualification.oracle",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(prefix) for prefix in forbidden):
                    return False
        if isinstance(node, ast.ImportFrom) and node.module:
            if any(node.module.startswith(prefix) for prefix in forbidden):
                return False
    return True


def run_qualification() -> QualificationReport:
    config = _config_from_env()
    client = LkwClient(config)
    try:
        client.wait_until_ready()
    except LkwClientError as exc:
        return _blocked_report(str(exc))

    records: list[QualificationRunRecord] = []
    try:
        for case in MANDATORY_CASES:
            records.append(_run_case(client, config, case))

        records.append(_run_case(client, config, Q2_F_HEALTHY, repeat_group="isolation"))
        records.append(_run_case(client, config, Q2_F_WRONG_TOOL, repeat_group="isolation"))

        repeat_records: list[QualificationRunRecord] = []
        for _ in range(3):
            repeat_records.append(_run_case(client, config, Q2_B_WRONG_TOOL, repeat_group=_REPEAT_CASE_ID))
        records.extend(repeat_records)
    except LkwClientError as exc:
        return _blocked_report(str(exc))

    repeatability_pass = len({_semantic_signature(item) for item in repeat_records}) == 1
    fidelity_pass = all(
        record.evidence_fidelity.candidate_fidelity_match
        and record.evidence_fidelity.selection_fidelity_match
        and record.evidence_fidelity.invocation_fidelity_match
        and record.evidence_fidelity.identity_fidelity_match
        for record in records
        if record.case_id != "Q2-E"
    )
    decision_independence_pass = _decision_diagnostics_independence_gate()

    matched = sum(1 for item in records if item.comparison.result is QualificationComparisonResult.MATCH)
    mismatched = len(records) - matched
    false_positives = sum(
        1
        for item in records
        if item.case_id in {"Q2-A", "Q2-F-A"}
        and item.comparison.result is QualificationComparisonResult.MISMATCH
        and any(
            mismatch.actual_status is FunctionalDiagnosticCheckStatus.PROVEN_FAIL
            for mismatch in item.comparison.check_mismatches
        )
    )
    false_negatives = sum(
        1
        for item in records
        if item.case_id in {"Q2-B", "Q2-C", "Q2-D", "Q2-F-B", _REPEAT_CASE_ID}
        and item.comparison.result is QualificationComparisonResult.MISMATCH
    )
    inconclusive_correct = sum(
        1
        for item in records
        if item.case_id == "Q2-E" and item.comparison.result is QualificationComparisonResult.MATCH
    )
    stage_accuracy = 100.0 if mismatched == 0 else (matched / len(records) * 100.0)
    inconclusive_accuracy = 100.0 if inconclusive_correct == 1 else 0.0

    verdict = (
        "PASS"
        if mismatched == 0
        and repeatability_pass
        and fidelity_pass
        and decision_independence_pass
        and false_positives == 0
        and false_negatives == 0
        else "FAILED"
    )
    report = QualificationReport(
        verdict=verdict,
        total_cases=len(records),
        matched_cases=matched,
        mismatched_cases=mismatched,
        false_positive_cases=false_positives,
        false_negative_cases=false_negatives,
        inconclusive_correct_cases=inconclusive_correct,
        stage_accuracy_percent=stage_accuracy,
        inconclusive_accuracy_percent=inconclusive_accuracy,
        repeatability_pass=repeatability_pass,
        records=tuple(records),
    )
    _write_artifact(
        report,
        fidelity_pass=fidelity_pass,
        decision_diagnostics_independence_pass=decision_independence_pass,
    )
    return report


def _blocked_report(reason: str) -> QualificationReport:
    report = QualificationReport(
        verdict="BLOCKED",
        total_cases=0,
        matched_cases=0,
        mismatched_cases=0,
        false_positive_cases=0,
        false_negative_cases=0,
        inconclusive_correct_cases=0,
        stage_accuracy_percent=0.0,
        inconclusive_accuracy_percent=0.0,
        repeatability_pass=False,
        records=(),
        blocked_reason=reason,
    )
    _write_artifact(report, fidelity_pass=False, decision_diagnostics_independence_pass=False)
    return report


def _write_artifact(
    report: QualificationReport,
    *,
    fidelity_pass: bool,
    decision_diagnostics_independence_pass: bool,
) -> None:
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "verdict": report.verdict,
        "blocked_reason": report.blocked_reason,
        "total_cases": report.total_cases,
        "matched_cases": report.matched_cases,
        "mismatched_cases": report.mismatched_cases,
        "false_positives": report.false_positive_cases,
        "false_negatives": report.false_negative_cases,
        "inconclusive_correct_cases": report.inconclusive_correct_cases,
        "stage_accuracy_percent": report.stage_accuracy_percent,
        "inconclusive_accuracy_percent": report.inconclusive_accuracy_percent,
        "repeatability_pass": report.repeatability_pass,
        "evidence_fidelity_pass": fidelity_pass,
        "decision_diagnostics_independence_pass": decision_diagnostics_independence_pass,
        "records": [
            {
                "case_id": record.case_id,
                "task_id": record.task_id,
                "run_id": record.run_id,
                "available_tools": list(record.available_tools),
                "expected_tool": record.expected_tool,
                "actual_tool": record.actual_tool,
                "invocation_status": record.invocation_status,
                "execution_outcome": record.execution_outcome.value,
                "functional_outcome": record.functional_outcome.value,
                "comparison_result": record.comparison.result.value,
                "diag_first_failed_check": record.diag_first_failed_check,
                "operator_outcome": record.operator_outcome,
                "repeat_group": record.repeat_group,
                "evidence_fidelity": {
                    "candidate_fidelity_match": record.evidence_fidelity.candidate_fidelity_match,
                    "selection_fidelity_match": record.evidence_fidelity.selection_fidelity_match,
                    "invocation_fidelity_match": record.evidence_fidelity.invocation_fidelity_match,
                    "validation_fidelity_match": record.evidence_fidelity.validation_fidelity_match,
                    "identity_fidelity_match": record.evidence_fidelity.identity_fidelity_match,
                },
            }
            for record in report.records
        ],
    }
    (_ARTIFACT_DIR / "qualification-report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    report = run_qualification()
    print(json.dumps({"verdict": report.verdict, "matched_cases": report.matched_cases}, indent=2))
    if report.verdict == "PASS":
        return 0
    if report.verdict == "BLOCKED":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
