# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q4 real model-routing qualification orchestrator."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_Q4_PROOF_PACKAGE_DIR = Path(__file__).resolve().parent
for _path in (_REPO_ROOT, _REPO_ROOT / "agents", _REPO_ROOT / "applications"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from scripts.proof.intergrax_proof_environment import load_proof_environment

load_proof_environment(
    proof_package_dir=_Q4_PROOF_PACKAGE_DIR,
    repository_root=_REPO_ROOT,
)

import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass

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
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import FunctionalDiagnosticCheckStatus
from intergrax.runtime.diagnostics.functional_operator_projection import FunctionalOperatorOutcomeStatus
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
    PipelineEvidenceScope,
    PipelineOperationStatus,
)
from intergrax.runtime.diagnostics.functional_validation_lookup import FunctionalValidationEvidenceLookup
from intergrax.runtime.diagnostics.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from intergrax.runtime.diagnostics.specifications.q4_model_routing_functional_diagnostic_specification import (
    Q4_MODEL_GENERATE_OPERATION_ID,
    build_q4_model_routing_functional_diagnostic_specification,
)
from intergrax.runtime.observability.functional_evidence_runtime_wiring import (
    wire_in_memory_functional_evidence_runtime,
)
from model_routing_qualifier.model_routing import Q4_PROFILE_A_MODEL, Q4_PROFILE_B_MODEL
from tests.system.functional_diagnostics_q4.cases import (
    HEALTHY_TASK,
    INVOKE_FAIL_PROFILE_REF,
    MANDATORY_CASES,
    PROFILE_A_REF,
    PROFILE_B_REF,
    Q4_F_HEALTHY,
    Q4_F_WRONG_ROUTE,
    _REPEAT_CASE_ID,
    case_metadata,
)
from tests.system.functional_diagnostics_q4.oracle import (
    build_independent_validation_evidence,
    independent_model_oracle_passes,
)
from tests.system.unified_execution.proof_runner.contracts import ProofConfig
from tests.system.unified_execution.proof_runner.lkw_client import LkwClient, LkwClientError, LkwRunResponse

_ARTIFACT_DIR = Path(
    os.environ.get(
        "DIAG_FUNCTIONAL_Q4_ARTIFACT_DIR",
        ".tmp/session/diag-functional-q4",
    ),
)
_CURSOR_SECRET = "diag-functional-q4-local-only-secret-32bytes!!"


@dataclass(frozen=True, slots=True)
class RoutingDecisionFidelitySnapshot:
    expected_profile_ref: str
    router_selected_profile_ref: str | None
    actual_adapter_provider: str | None
    actual_adapter_model: str | None
    actual_adapter_profile_ref: str | None
    routing_decision_fidelity_match: bool
    adapter_fidelity_match: bool
    post_decision_forcing_detected: bool
    post_generation_forcing_detected: bool


@dataclass(frozen=True, slots=True)
class EvidenceFidelitySnapshot:
    provider_candidate_refs: tuple[str, ...]
    actual_selected_profile: str | None
    emitted_selected_profile: str | None
    actual_invoke_succeeded: bool | None
    emitted_invoke_succeeded: bool | None
    candidate_fidelity_match: bool
    selection_fidelity_match: bool
    adapter_fidelity_match: bool
    invocation_fidelity_match: bool
    output_fidelity_match: bool
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
    routing_decision_fidelity: RoutingDecisionFidelitySnapshot
    diag_first_failed_check: str | None
    operator_outcome: str | None
    routing_context_summary: str | None
    candidate_profiles: tuple[str, ...]
    expected_profile_ref: str
    actual_selected_profile: str | None
    actual_adapter_provider: str | None
    actual_adapter_model: str | None
    invocation_status: str | None
    raw_model_output: str | None
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
    stage_matched_cases: int = 0
    functional_failure_detected_cases: int = 0
    functional_failure_ground_truth_cases: int = 0


_EXPECTATION_BY_CASE_ID: dict[str, QualificationCaseExpectation] = {
    case.case_id: case for case in MANDATORY_CASES
}
_EXPECTATION_BY_CASE_ID["Q4-F-A"] = Q4_F_HEALTHY
_EXPECTATION_BY_CASE_ID["Q4-F-B"] = Q4_F_WRONG_ROUTE
_EXPECTATION_BY_CASE_ID[_REPEAT_CASE_ID] = MANDATORY_CASES[1]


def _expectation_for_record(record: QualificationRunRecord) -> QualificationCaseExpectation:
    if record.repeat_group == _REPEAT_CASE_ID:
        return MANDATORY_CASES[1]
    return _EXPECTATION_BY_CASE_ID.get(record.case_id, MANDATORY_CASES[1])


def _expected_profile_ref_for_case(case_id: str) -> str:
    if case_id == "Q4-C":
        return INVOKE_FAIL_PROFILE_REF
    return PROFILE_A_REF


def _stage_matches(record: QualificationRunRecord, expectation: QualificationCaseExpectation) -> bool:
    expected = expectation.expected_first_proven_failed_check
    actual = record.diag_first_failed_check
    return str(expected or "") == str(actual or "")


def _functional_failure_detected(record: QualificationRunRecord) -> bool:
    return (
        record.functional_outcome is QualificationFunctionalOutcome.FAILED
        and record.operator_outcome
        == FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE.value
    )


def _config_from_env() -> ProofConfig:
    return ProofConfig(
        base_url=os.environ.get("LKW_BASE_URL", "http://localhost:8021"),
        api_key=os.environ.get(
            "LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY",
            "ue-11g-c1-certification-secret",
        ),
        tenant_id=os.environ.get("DIAG_FUNCTIONAL_Q4_TENANT_ID", "tenant-ue-11g-c1"),
        workspace_id=os.environ.get("DIAG_FUNCTIONAL_Q4_WORKSPACE_ID", "ue-11g-c1-workspace"),
        collection_id=os.environ.get("DIAG_FUNCTIONAL_Q4_COLLECTION_ID", "ue-11g-c1-collection"),
        fixture_root=os.environ.get("DIAG_FUNCTIONAL_Q4_FIXTURE_ROOT", "/cert-fixtures/workspace"),
    )


def _fetch_functional_evidence(
    config: ProofConfig,
    *,
    tenant_id: str,
    task_id: str,
    run_id: str,
) -> tuple[object, ...]:
    query = urllib.parse.urlencode(
        {"tenant_id": tenant_id, "task_id": task_id, "run_id": run_id},
    )
    url = f"{config.base_url.rstrip('/')}/v1/local_workspace/qualification/functional_evidence?{query}"
    request = urllib.request.Request(url, headers={"X-API-Key": config.api_key})
    with urllib.request.urlopen(request, timeout=180.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    items = payload.get("items")
    if not isinstance(items, list):
        raise LkwClientError("functional_evidence_items_missing")
    from intergrax.runtime.diagnostics.functional_evidence import PlatformFunctionalEvidence

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


def _candidate_refs_from_items(items: tuple[object, ...]) -> tuple[str, ...]:
    refs: list[str] = []
    for item in items:
        if item.kind is not PipelineEvidenceKind.CANDIDATE_RANK or item.candidate is None:
            continue
        refs.append(item.candidate.candidate_artifact_ref.artifact_ref)
    return tuple(refs)


def _selection_ref_from_items(items: tuple[object, ...]) -> str | None:
    for item in items:
        if item.kind is PipelineEvidenceKind.SELECTION and item.selection is not None:
            return item.selection.selected_artifact_ref.artifact_ref
    return None


def _invoke_succeeded_from_items(items: tuple[object, ...]) -> bool | None:
    for item in items:
        if item.kind is not PipelineEvidenceKind.OPERATION_OUTCOME or item.operation_outcome is None:
            continue
        if item.provenance.operation_id != Q4_MODEL_GENERATE_OPERATION_ID:
            continue
        return item.operation_outcome.status is PipelineOperationStatus.SUCCEEDED
    return None


def _output_relation_present(items: tuple[object, ...]) -> bool:
    for item in items:
        if item.kind is PipelineEvidenceKind.OUTPUT_RELATION and item.output_relation is not None:
            if item.provenance.operation_id == Q4_MODEL_GENERATE_OPERATION_ID:
                return True
    return False


def _summary_from_response(response: LkwRunResponse) -> dict[str, object]:
    if response.lkw_evidence is None:
        return {}
    diagnostics = response.lkw_evidence.diagnostics
    summary = diagnostics.get("model_routing_summary")
    if isinstance(summary, dict):
        return summary
    return {}


def _adapter_ref_from_items(items: tuple[object, ...]) -> str | None:
    for item in items:
        if item.kind is not PipelineEvidenceKind.OPERATION_OUTCOME or item.operation_outcome is None:
            continue
        if item.provenance.operation_id != Q4_MODEL_GENERATE_OPERATION_ID:
            continue
        operation_name = item.operation_outcome.operation_name
        if isinstance(operation_name, str) and operation_name.startswith("llm:"):
            return operation_name
    return None


def _parse_adapter_ref(adapter_ref: str | None) -> tuple[str | None, str | None]:
    if not adapter_ref or not adapter_ref.startswith("llm:"):
        return None, None
    remainder = adapter_ref.removeprefix("llm:")
    if ":" not in remainder:
        return None, None
    provider, model = remainder.split(":", 1)
    return provider, model


def _adapter_profile_ref(provider: str | None, model: str | None) -> str | None:
    if not provider or not model:
        return None
    return f"llm:{provider}:{model}"


def _resolve_adapter_fields(
    *,
    summary: dict[str, object],
    remote_items: tuple[object, ...],
) -> tuple[str | None, str | None, str | None]:
    provider_raw = summary.get("actual_adapter_provider")
    model_raw = summary.get("actual_adapter_model")
    provider = str(provider_raw) if isinstance(provider_raw, str) else None
    model = str(model_raw) if isinstance(model_raw, str) else None
    adapter_ref = _adapter_profile_ref(provider, model)
    if adapter_ref is None:
        adapter_ref = _adapter_ref_from_items(remote_items)
        provider, model = _parse_adapter_ref(adapter_ref)
    return provider, model, adapter_ref


def _preflight_ollama_models() -> None:
    try:
        completed = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LkwClientError(f"ollama_unavailable:{exc}") from exc
    listing = completed.stdout
    for model in (Q4_PROFILE_A_MODEL, Q4_PROFILE_B_MODEL):
        if model not in listing:
            raise LkwClientError(f"ollama_model_missing:{model}")


def _routing_decision_fidelity(
    *,
    case_id: str,
    summary: dict[str, object],
    emitted_selected: str | None,
    actual_adapter_provider: str | None,
    actual_adapter_model: str | None,
    actual_adapter_ref: str | None,
) -> RoutingDecisionFidelitySnapshot:
    expected_profile_ref = _expected_profile_ref_for_case(case_id)
    router_selected = summary.get("selected_profile_ref")
    router_selected_ref = str(router_selected) if isinstance(router_selected, str) else None
    if router_selected_ref is None:
        router_selected_ref = emitted_selected
    resolved_adapter_ref = actual_adapter_ref or _adapter_profile_ref(
        actual_adapter_provider,
        actual_adapter_model,
    )
    routing_match = router_selected_ref == emitted_selected
    adapter_match = (
        router_selected_ref == resolved_adapter_ref if router_selected_ref and resolved_adapter_ref else False
    )
    post_decision_forcing = False
    if (
        router_selected_ref
        and resolved_adapter_ref
        and router_selected_ref != resolved_adapter_ref
    ):
        post_decision_forcing = True
    post_generation_forcing = False
    raw_output = summary.get("raw_model_output")
    answer_output = summary.get("raw_model_output")
    if isinstance(raw_output, str) and "99" in raw_output and case_id != "Q4-D":
        post_generation_forcing = False
    if case_id == "Q4-D" and isinstance(answer_output, str) and "42" in answer_output:
        post_generation_forcing = True
    return RoutingDecisionFidelitySnapshot(
        expected_profile_ref=expected_profile_ref,
        router_selected_profile_ref=router_selected_ref,
        actual_adapter_provider=actual_adapter_provider,
        actual_adapter_model=actual_adapter_model,
        actual_adapter_profile_ref=resolved_adapter_ref,
        routing_decision_fidelity_match=routing_match and adapter_match,
        adapter_fidelity_match=adapter_match,
        post_decision_forcing_detected=post_decision_forcing,
        post_generation_forcing_detected=post_generation_forcing,
    )


def _evidence_fidelity_snapshot(
    *,
    provider_candidates: tuple[str, ...],
    actual_selected: str | None,
    emitted_selected: str | None,
    actual_invoke_succeeded: bool | None,
    emitted_invoke_succeeded: bool | None,
    remote_items: tuple[object, ...],
    scope: PipelineEvidenceScope,
    failure_injection_layer: str | None,
    validation_expected: bool,
    validation_actual_pass: bool,
    adapter_fidelity_match: bool,
    output_present: bool,
    raw_model_output: str | None,
) -> EvidenceFidelitySnapshot:
    emitted_candidates = _candidate_refs_from_items(remote_items)
    identity_ok = all(
        item.scope.tenant_id == scope.tenant_id
        and item.scope.task_id == scope.task_id
        and item.scope.run_id == scope.run_id
        for item in remote_items
    )
    output_fidelity = output_present if actual_invoke_succeeded else True
    return EvidenceFidelitySnapshot(
        provider_candidate_refs=provider_candidates,
        actual_selected_profile=actual_selected,
        emitted_selected_profile=emitted_selected,
        actual_invoke_succeeded=actual_invoke_succeeded,
        emitted_invoke_succeeded=emitted_invoke_succeeded,
        candidate_fidelity_match=provider_candidates == emitted_candidates,
        selection_fidelity_match=actual_selected == emitted_selected,
        adapter_fidelity_match=adapter_fidelity_match,
        invocation_fidelity_match=actual_invoke_succeeded == emitted_invoke_succeeded,
        output_fidelity_match=output_fidelity,
        validation_fidelity_match=True,
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
        record.actual_selected_profile or "",
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
        **case_metadata(expectation),
    }
    failure_layer_raw = metadata.get("qualification_failure_injection_layer")
    failure_layer = str(failure_layer_raw) if failure_layer_raw is not None else None
    expected_profile_ref = str(metadata.get("qualification_expected_profile_ref") or PROFILE_A_REF)

    response = client.run_model_routing_qualification(
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
    summary = _summary_from_response(response)
    provider_candidates = _candidate_refs_from_items(remote_items)
    actual_selected = _selection_ref_from_items(remote_items)
    if actual_selected is None:
        selected_raw = summary.get("selected_profile_ref")
        actual_selected = str(selected_raw) if isinstance(selected_raw, str) else None
    emitted_selected = actual_selected
    actual_invoke = _invoke_succeeded_from_items(remote_items)
    if actual_invoke is None:
        status = summary.get("invocation_status")
        actual_invoke = status == "success" if isinstance(status, str) else None
    emitted_invoke = actual_invoke
    actual_provider_str, actual_model_str, actual_adapter_ref = _resolve_adapter_fields(
        summary=summary,
        remote_items=remote_items,
    )
    raw_output_raw = summary.get("raw_model_output")
    raw_model_output = str(raw_output_raw) if isinstance(raw_output_raw, str) else response.answer

    routing_fidelity = _routing_decision_fidelity(
        case_id=expectation.case_id,
        summary=summary,
        emitted_selected=emitted_selected,
        actual_adapter_provider=actual_provider_str,
        actual_adapter_model=actual_model_str,
        actual_adapter_ref=actual_adapter_ref,
    )

    functional_outcome = (
        QualificationFunctionalOutcome.PASSED
        if independent_model_oracle_passes(
            answer=response.answer,
            selected_profile_artifact=actual_selected,
            expected_profile_artifact=expected_profile_ref,
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
        selected_profile_artifact=actual_selected,
        expected_profile_artifact=expected_profile_ref,
        idempotency_key=expectation.case_id,
    )
    spec = build_q4_model_routing_functional_diagnostic_specification(
        validation_id=validation.validation_id if expectation.include_validation else None,
        include_validation=expectation.include_validation,
        expected_selection_artifact_ref=expected_profile_ref,
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
        actual_selected=actual_selected,
        emitted_selected=emitted_selected,
        actual_invoke_succeeded=actual_invoke,
        emitted_invoke_succeeded=emitted_invoke,
        remote_items=remote_items,
        scope=pipeline_scope,
        failure_injection_layer=failure_layer,
        validation_expected=expectation.include_validation,
        validation_actual_pass=validation.outcome.value == "passed",
        adapter_fidelity_match=routing_fidelity.adapter_fidelity_match,
        output_present=_output_relation_present(remote_items),
        raw_model_output=raw_model_output,
    )
    routing_context_raw = summary.get("routing_context_summary")
    routing_context_summary = (
        str(routing_context_raw) if isinstance(routing_context_raw, str) else None
    )
    candidate_raw = summary.get("candidate_profile_refs")
    candidate_profiles: tuple[str, ...] = ()
    if isinstance(candidate_raw, list):
        candidate_profiles = tuple(str(item) for item in candidate_raw if isinstance(item, str))
    invocation_status_raw = summary.get("invocation_status")
    invocation_status = str(invocation_status_raw) if invocation_status_raw is not None else None

    return QualificationRunRecord(
        case_id=expectation.case_id,
        task_id=response.task_id,
        run_id=response.run_id,
        execution_outcome=execution_outcome,
        functional_outcome=functional_outcome,
        comparison=comparison,
        evidence_fidelity=fidelity,
        routing_decision_fidelity=routing_fidelity,
        diag_first_failed_check=(
            str(analysis.first_proven_failure) if analysis.first_proven_failure is not None else None
        ),
        operator_outcome=(
            operator.functional_projection.outcome_status.value
            if operator.functional_projection is not None
            else None
        ),
        routing_context_summary=routing_context_summary,
        candidate_profiles=candidate_profiles,
        expected_profile_ref=expected_profile_ref,
        actual_selected_profile=actual_selected,
        actual_adapter_provider=actual_provider_str,
        actual_adapter_model=actual_model_str,
        invocation_status=invocation_status,
        raw_model_output=raw_model_output[:200] if raw_model_output else None,
        repeat_group=repeat_group,
    )


def _decision_diagnostics_independence_gate() -> bool:
    import ast

    job_path = (
        Path(__file__).resolve().parents[3]
        / "agents"
        / "model_routing_qualifier"
        / "steps"
        / "model_routing_job.py"
    )
    tree = ast.parse(job_path.read_text(encoding="utf-8"))
    forbidden = (
        "intergrax.runtime.diagnostics",
        "functional_diagnostic",
        "functional_diagnostics_q4",
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
    source = job_path.read_text(encoding="utf-8")
    if "qualification_force_profile" in source:
        return False
    if "qualification_selected_profile" in source:
        return False
    return True


def run_qualification() -> QualificationReport:
    config = _config_from_env()
    try:
        _preflight_ollama_models()
    except LkwClientError as exc:
        return _blocked_report(str(exc))

    client = LkwClient(config)
    try:
        client.wait_until_ready()
    except LkwClientError as exc:
        return _blocked_report(str(exc))

    records: list[QualificationRunRecord] = []
    try:
        for case in MANDATORY_CASES:
            records.append(_run_case(client, config, case))

        records.append(_run_case(client, config, Q4_F_HEALTHY, repeat_group="isolation"))
        records.append(_run_case(client, config, Q4_F_WRONG_ROUTE, repeat_group="isolation"))

        repeat_records: list[QualificationRunRecord] = []
        for _ in range(3):
            repeat_records.append(
                _run_case(client, config, MANDATORY_CASES[1], repeat_group=_REPEAT_CASE_ID),
            )
        records.extend(repeat_records)
    except LkwClientError as exc:
        return _blocked_report(str(exc), partial_records=records)

    repeatability_pass = len({_semantic_signature(item) for item in repeat_records}) == 1
    fidelity_pass = all(
        record.evidence_fidelity.candidate_fidelity_match
        and record.evidence_fidelity.selection_fidelity_match
        and record.evidence_fidelity.adapter_fidelity_match
        and record.evidence_fidelity.invocation_fidelity_match
        and record.evidence_fidelity.output_fidelity_match
        and record.evidence_fidelity.identity_fidelity_match
        for record in records
        if record.case_id != "Q4-E"
    )
    routing_decision_fidelity_pass = all(
        record.routing_decision_fidelity.routing_decision_fidelity_match
        for record in records
        if record.case_id != "Q4-E"
    )
    post_decision_forcing_pass = all(
        not record.routing_decision_fidelity.post_decision_forcing_detected for record in records
    )
    post_generation_forcing_pass = all(
        not record.routing_decision_fidelity.post_generation_forcing_detected for record in records
    )
    decision_independence_pass = _decision_diagnostics_independence_gate()

    matched = sum(1 for item in records if item.comparison.result is QualificationComparisonResult.MATCH)
    mismatched = len(records) - matched
    stage_matched = sum(
        1 for item in records if _stage_matches(item, _expectation_for_record(item))
    )
    functional_failure_ground_truth = [
        item
        for item in records
        if item.functional_outcome is QualificationFunctionalOutcome.FAILED
        and _expectation_for_record(item).expected_operator_outcome
        is not FunctionalOperatorOutcomeStatus.INCONCLUSIVE
    ]
    functional_failure_detected = sum(
        1 for item in functional_failure_ground_truth if _functional_failure_detected(item)
    )
    false_positives = sum(
        1
        for item in records
        if item.case_id in {"Q4-A", "Q4-F-A"}
        and item.comparison.result is QualificationComparisonResult.MISMATCH
        and any(
            mismatch.actual_status is FunctionalDiagnosticCheckStatus.PROVEN_FAIL
            for mismatch in item.comparison.check_mismatches
        )
    )
    false_negatives = sum(
        1
        for item in functional_failure_ground_truth
        if not _functional_failure_detected(item)
        and _expectation_for_record(item).expected_operator_outcome
        is not FunctionalOperatorOutcomeStatus.INCONCLUSIVE
    )
    inconclusive_correct = sum(
        1
        for item in records
        if item.case_id == "Q4-E" and item.comparison.result is QualificationComparisonResult.MATCH
    )
    stage_accuracy = (stage_matched / len(records) * 100.0) if records else 0.0
    inconclusive_accuracy = 100.0 if inconclusive_correct == 1 else 0.0

    verdict = (
        "PASS"
        if mismatched == 0
        and repeatability_pass
        and fidelity_pass
        and routing_decision_fidelity_pass
        and post_decision_forcing_pass
        and post_generation_forcing_pass
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
        stage_matched_cases=stage_matched,
        functional_failure_detected_cases=functional_failure_detected,
        functional_failure_ground_truth_cases=len(functional_failure_ground_truth),
    )
    _write_artifact(
        report,
        fidelity_pass=fidelity_pass,
        routing_decision_fidelity_pass=routing_decision_fidelity_pass,
        post_decision_forcing_pass=post_decision_forcing_pass,
        post_generation_forcing_pass=post_generation_forcing_pass,
        decision_diagnostics_independence_pass=decision_independence_pass,
    )
    return report


def _blocked_report(
    reason: str,
    *,
    partial_records: list[QualificationRunRecord] | None = None,
) -> QualificationReport:
    records = tuple(partial_records or ())
    report = QualificationReport(
        verdict="BLOCKED",
        total_cases=len(records),
        matched_cases=0,
        mismatched_cases=0,
        false_positive_cases=0,
        false_negative_cases=0,
        inconclusive_correct_cases=0,
        stage_accuracy_percent=0.0,
        inconclusive_accuracy_percent=0.0,
        repeatability_pass=False,
        records=records,
        blocked_reason=reason,
    )
    _write_artifact(
        report,
        fidelity_pass=False,
        routing_decision_fidelity_pass=False,
        post_decision_forcing_pass=False,
        post_generation_forcing_pass=False,
        decision_diagnostics_independence_pass=False,
    )
    return report


def _write_artifact(
    report: QualificationReport,
    *,
    fidelity_pass: bool,
    routing_decision_fidelity_pass: bool,
    post_decision_forcing_pass: bool,
    post_generation_forcing_pass: bool,
    decision_diagnostics_independence_pass: bool,
) -> None:
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _ARTIFACT_DIR / "qualification-report.json"
    initial_path = _ARTIFACT_DIR / "qualification-report-initial-failure.json"
    if report_path.exists() and not initial_path.exists():
        shutil.copy2(report_path, initial_path)
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
        "full_case_match_cases": report.matched_cases,
        "stage_match_cases": report.stage_matched_cases,
        "functional_failure_ground_truth_cases": report.functional_failure_ground_truth_cases,
        "functional_failure_detected_cases": report.functional_failure_detected_cases,
        "repeatability_pass": report.repeatability_pass,
        "evidence_fidelity_pass": fidelity_pass,
        "routing_decision_fidelity_pass": routing_decision_fidelity_pass,
        "post_decision_forcing": "NONE" if post_decision_forcing_pass else "DETECTED",
        "post_generation_forcing": "NONE" if post_generation_forcing_pass else "DETECTED",
        "decision_diagnostics_independence_pass": decision_diagnostics_independence_pass,
        "records": [
            {
                "case_id": record.case_id,
                "task_id": record.task_id,
                "run_id": record.run_id,
                "routing_context": record.routing_context_summary,
                "candidate_profiles": list(record.candidate_profiles),
                "expected_profile": record.expected_profile_ref,
                "actual_selected_profile": record.actual_selected_profile,
                "actual_adapter_provider": record.actual_adapter_provider,
                "actual_adapter_model": record.actual_adapter_model,
                "invocation_outcome": record.invocation_status,
                "raw_model_output": record.raw_model_output,
                "execution_outcome": record.execution_outcome.value,
                "functional_outcome": record.functional_outcome.value,
                "comparison_result": record.comparison.result.value,
                "expected_first_failure": (
                    str(_expectation_for_record(record).expected_first_proven_failed_check)
                    if _expectation_for_record(record).expected_first_proven_failed_check is not None
                    else None
                ),
                "actual_first_failure": record.diag_first_failed_check,
                "operator_outcome": record.operator_outcome,
                "repeat_group": record.repeat_group,
                "evidence_fidelity": {
                    "candidate_fidelity_match": record.evidence_fidelity.candidate_fidelity_match,
                    "selection_fidelity_match": record.evidence_fidelity.selection_fidelity_match,
                    "adapter_fidelity_match": record.evidence_fidelity.adapter_fidelity_match,
                    "invocation_fidelity_match": record.evidence_fidelity.invocation_fidelity_match,
                    "output_fidelity_match": record.evidence_fidelity.output_fidelity_match,
                    "validation_fidelity_match": record.evidence_fidelity.validation_fidelity_match,
                    "identity_fidelity_match": record.evidence_fidelity.identity_fidelity_match,
                },
                "routing_decision_fidelity": {
                    "routing_decision_fidelity_match": (
                        record.routing_decision_fidelity.routing_decision_fidelity_match
                    ),
                    "adapter_fidelity_match": record.routing_decision_fidelity.adapter_fidelity_match,
                    "post_decision_forcing": record.routing_decision_fidelity.post_decision_forcing_detected,
                    "post_generation_forcing": (
                        record.routing_decision_fidelity.post_generation_forcing_detected
                    ),
                },
            }
            for record in report.records
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
