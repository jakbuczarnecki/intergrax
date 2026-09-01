# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q1 real RAG/C1 qualification orchestrator."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intergrax.contracts.execution_identity import validate_run_id, validate_task_id
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
    PlatformFunctionalEvidence,
)
from intergrax.runtime.diagnostics.functional_validation_lookup import FunctionalValidationEvidenceLookup
from intergrax.runtime.diagnostics.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from intergrax.runtime.diagnostics.specifications.c1_rag_functional_diagnostic_specification import (
    build_c1_rag_functional_diagnostic_specification,
)
from intergrax.runtime.observability.functional_evidence_runtime_wiring import (
    wire_in_memory_functional_evidence_runtime,
)
from tests.system.functional_diagnostics_q1.cases import (
    MANDATORY_CASES,
    Q1_B_SELECTION_FAILURE,
    Q1_E_FAILURE,
    Q1_E_HEALTHY,
    case_metadata,
)
from tests.system.functional_diagnostics_q1.oracle import (
    build_independent_validation_evidence,
    evidence_texts_from_lkw_response,
    resolve_qualification_functional_pass,
    search_request_message,
)
from tests.system.unified_execution.proof_runner.contracts import ProofConfig
from tests.system.unified_execution.proof_runner.lkw_client import LkwClient, LkwClientError, LkwRunResponse

_ARTIFACT_DIR = Path(
    os.environ.get(
        "DIAG_FUNCTIONAL_Q1_ARTIFACT_DIR",
        ".tmp/session/diag-functional-q1",
    ),
)
_FIXTURE_FILES = (
    "architecture.md",
    "incident-report.md",
    "operations.md",
    "operations-decoy.md",
)
_DOCKER_FIXTURE_ROOT = "/cert-fixtures/workspace"
_REPEAT_CASE_ID = "Q1-F"
_CURSOR_SECRET = "diag-functional-q1-local-only-secret-32bytes!!"


@dataclass(frozen=True, slots=True)
class QualificationRunRecord:
    case_id: str
    task_id: str
    run_id: str
    execution_outcome: QualificationExecutionOutcome
    functional_outcome: QualificationFunctionalOutcome
    comparison: QualificationCaseComparison
    evidence_fidelity_ok: bool
    diag_first_failed_check: str | None
    operator_outcome: str | None
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
        tenant_id=os.environ.get("DIAG_FUNCTIONAL_Q1_TENANT_ID", "tenant-ue-11g-c1"),
        workspace_id=os.environ.get("DIAG_FUNCTIONAL_Q1_WORKSPACE_ID", "ue-11g-c1-workspace"),
        collection_id=os.environ.get("DIAG_FUNCTIONAL_Q1_COLLECTION_ID", "ue-11g-c1-collection"),
        fixture_root=os.environ.get("DIAG_FUNCTIONAL_Q1_FIXTURE_ROOT", _DOCKER_FIXTURE_ROOT),
    )


def _fixture_paths(config: ProofConfig) -> list[str]:
    root = config.fixture_root.rstrip("/\\")
    if root == _DOCKER_FIXTURE_ROOT:
        return [f"{_DOCKER_FIXTURE_ROOT}/{name}" for name in _FIXTURE_FILES]
    base = Path(root)
    return [str((base / name).resolve()) for name in _FIXTURE_FILES]


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


def _replay_into_persistence(
    persistence: InMemoryFunctionalEvidencePersistence,
    items: tuple[PlatformFunctionalEvidence, ...],
) -> None:
    for item in items:
        persistence.append(item)


def _semantic_signature(record: QualificationRunRecord) -> tuple[str, str | None, str | None]:
    failed_checks = tuple(
        f"{m.check_id}:{m.actual_status.value}"
        for m in record.comparison.check_mismatches
    )
    return (
        record.comparison.result.value,
        record.diag_first_failed_check,
        record.operator_outcome,
    ) if not failed_checks else (
        record.comparison.result.value,
        record.diag_first_failed_check,
        ",".join(
            f"{item.check_id}:{item.expected_status.value}->{item.actual_status.value}"
            for item in record.comparison.check_mismatches
        ),
    )


def _resope_evidence_items(
    items: tuple[PlatformFunctionalEvidence, ...],
    *,
    tenant_id: str,
    task_id: str,
    run_id: str,
) -> tuple[PlatformFunctionalEvidence, ...]:
    scope = PipelineEvidenceScope(
        tenant_id=tenant_id,
        task_id=validate_task_id(task_id),
        run_id=validate_run_id(run_id),
    )
    return tuple(item.model_copy(update={"scope": scope}) for item in items)


def _run_synthesis_case(
    client: LkwClient,
    config: ProofConfig,
    metadata: dict[str, object],
) -> tuple[LkwRunResponse, LkwRunResponse, tuple[PlatformFunctionalEvidence, ...]]:
    search_response = client.run_search(
        message=search_request_message(),
        metadata=metadata,
    )
    search_remote = _fetch_functional_evidence(
        config,
        tenant_id=config.tenant_id,
        task_id=search_response.task_id,
        run_id=search_response.run_id,
    )
    synth_metadata = {
        **metadata,
        "shadow_workspace": True,
        "output_name": metadata.get("output_name", "q1-synthesis-draft.md"),
    }
    synth_response = client.run_synthesize(
        message=search_request_message(),
        metadata=synth_metadata,
    )
    synth_remote = _fetch_functional_evidence(
        config,
        tenant_id=config.tenant_id,
        task_id=synth_response.task_id,
        run_id=synth_response.run_id,
    )
    merged = search_remote + _resope_evidence_items(
        synth_remote,
        tenant_id=config.tenant_id,
        task_id=search_response.task_id,
        run_id=search_response.run_id,
    )
    return search_response, synth_response, merged


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
        "collection_id": config.collection_id,
        "query": search_request_message(),
        "top_k": 5,
        **case_metadata(expectation),
    }
    remote_items: tuple[PlatformFunctionalEvidence, ...] | None = None
    if expectation.include_output_relation:
        scope_response, response, remote_items = _run_synthesis_case(client, config, metadata)
    else:
        response = client.run_search(
            message=search_request_message(),
            metadata=metadata,
        )
        scope_response = response

    execution_outcome = (
        QualificationExecutionOutcome.COMPLETED
        if response.state == "completed"
        else QualificationExecutionOutcome.FAILED
    )
    lkw_evidence_dict = response.lkw_evidence.model_dump() if response.lkw_evidence is not None else None
    evidence_texts = evidence_texts_from_lkw_response(
        answer=response.answer,
        lkw_evidence=lkw_evidence_dict,
    )
    draft_override = metadata.get("qualification_draft_override")
    if isinstance(draft_override, str) and draft_override.strip():
        evidence_texts.append(draft_override.strip())
    functional_outcome = (
        QualificationFunctionalOutcome.PASSED
        if resolve_qualification_functional_pass(
            metadata=metadata,
            answer=response.answer,
            evidence_texts=evidence_texts,
        )
        else QualificationFunctionalOutcome.FAILED
    )

    if remote_items is None:
        remote_items = _fetch_functional_evidence(
            config,
            tenant_id=config.tenant_id,
            task_id=response.task_id,
            run_id=response.run_id,
        )
    wiring = wire_in_memory_functional_evidence_runtime(cursor_secret=_CURSOR_SECRET)
    _replay_into_persistence(wiring.persistence, remote_items)

    pipeline_scope = PipelineEvidenceScope(
        tenant_id=config.tenant_id,
        task_id=validate_task_id(scope_response.task_id),
        run_id=validate_run_id(scope_response.run_id),
    )
    validation = build_independent_validation_evidence(
        pipeline_scope,
        answer=response.answer,
        evidence_texts=evidence_texts,
        idempotency_key=expectation.case_id,
        metadata=metadata,
    )
    spec = build_c1_rag_functional_diagnostic_specification(
        validation_id=validation.validation_id if expectation.include_validation else None,
        include_output_relation=expectation.include_output_relation,
        include_validation=expectation.include_validation,
    )
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
    operator = DiagnosticAssessmentComposer().compose(lifecycle, analysis)
    comparison = compare_qualification_case(
        expectation,
        actual_execution_outcome=execution_outcome,
        actual_functional_outcome=functional_outcome,
        analysis=analysis,
        operator_assessment=operator,
    )
    fidelity_ok = any(item.kind is PipelineEvidenceKind.OPERATION_OUTCOME for item in remote_items)
    return QualificationRunRecord(
        case_id=expectation.case_id,
        task_id=scope_response.task_id,
        run_id=scope_response.run_id,
        execution_outcome=execution_outcome,
        functional_outcome=functional_outcome,
        comparison=comparison,
        evidence_fidelity_ok=fidelity_ok,
        diag_first_failed_check=(
            str(analysis.first_proven_failure) if analysis.first_proven_failure is not None else None
        ),
        operator_outcome=(
            operator.functional_projection.outcome_status.value
            if operator.functional_projection is not None
            else None
        ),
        repeat_group=repeat_group,
    )


def run_qualification() -> QualificationReport:
    config = _config_from_env()
    client = LkwClient(config)
    try:
        client.wait_until_ready()
    except LkwClientError as exc:
        return _blocked_report(str(exc))

    index_response = client.run_index(source_paths=_fixture_paths(config))
    if index_response.state != "completed":
        return _blocked_report(f"index_state_{index_response.state}")

    records: list[QualificationRunRecord] = []
    for case in MANDATORY_CASES:
        records.append(_run_case(client, config, case))

    records.append(_run_case(client, config, Q1_E_HEALTHY, repeat_group="isolation"))
    records.append(_run_case(client, config, Q1_E_FAILURE, repeat_group="isolation"))

    repeat_records: list[QualificationRunRecord] = []
    for _ in range(3):
        repeat_records.append(
            _run_case(
                client,
                config,
                Q1_B_SELECTION_FAILURE,
                repeat_group=_REPEAT_CASE_ID,
            ),
        )
    records.extend(repeat_records)
    repeatability_pass = len({_semantic_signature(item) for item in repeat_records}) == 1

    matched = sum(1 for item in records if item.comparison.result is QualificationComparisonResult.MATCH)
    mismatched = len(records) - matched
    false_positives = sum(
        1
        for item in records
        if item.case_id in {"Q1-A", "Q1-E-A"}
        and item.comparison.result is QualificationComparisonResult.MISMATCH
        and any(
            mismatch.actual_status is FunctionalDiagnosticCheckStatus.PROVEN_FAIL
            for mismatch in item.comparison.check_mismatches
        )
    )
    false_negatives = sum(
        1
        for item in records
        if item.case_id in {"Q1-B", "Q1-C", "Q1-E-B", "Q1-F"}
        and item.comparison.result is QualificationComparisonResult.MISMATCH
    )
    inconclusive_correct = sum(
        1
        for item in records
        if item.case_id == "Q1-D" and item.comparison.result is QualificationComparisonResult.MATCH
    )

    all_matched = mismatched == 0
    verdict = "PASS" if all_matched and repeatability_pass else "FAILED"
    report = QualificationReport(
        verdict=verdict,
        total_cases=len(records),
        matched_cases=matched,
        mismatched_cases=mismatched,
        false_positive_cases=false_positives,
        false_negative_cases=false_negatives,
        inconclusive_correct_cases=inconclusive_correct,
        repeatability_pass=repeatability_pass,
        records=tuple(records),
    )
    _write_artifact(report)
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
        repeatability_pass=False,
        records=(),
        blocked_reason=reason,
    )
    _write_artifact(report)
    return report


def _write_artifact(report: QualificationReport) -> None:
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "verdict": report.verdict,
        "total_cases": report.total_cases,
        "matched_cases": report.matched_cases,
        "mismatched_cases": report.mismatched_cases,
        "false_positive_cases": report.false_positive_cases,
        "false_negative_cases": report.false_negative_cases,
        "inconclusive_correct_cases": report.inconclusive_correct_cases,
        "repeatability_pass": report.repeatability_pass,
        "blocked_reason": report.blocked_reason,
        "records": [
            {
                "case_id": item.case_id,
                "task_id": item.task_id,
                "run_id": item.run_id,
                "execution_outcome": item.execution_outcome.value,
                "functional_outcome": item.functional_outcome.value,
                "comparison": item.comparison.result.value,
                "diag_first_failed_check": item.diag_first_failed_check,
                "operator_outcome": item.operator_outcome,
                "evidence_fidelity_ok": item.evidence_fidelity_ok,
                "repeat_group": item.repeat_group,
                "check_mismatches": [
                    {
                        "check_id": str(m.check_id),
                        "expected": m.expected_status.value,
                        "actual": m.actual_status.value,
                    }
                    for m in item.comparison.check_mismatches
                ],
                "field_mismatches": [
                    {"field": m.field, "expected": m.expected, "actual": m.actual}
                    for m in item.comparison.field_mismatches
                ],
            }
            for item in report.records
        ],
    }
    (_ARTIFACT_DIR / "qualification-report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    result = run_qualification()
    print(json.dumps({"verdict": result.verdict, "matched": result.matched_cases, "total": result.total_cases}))
    raise SystemExit(0 if result.verdict == "PASS" else 1)
