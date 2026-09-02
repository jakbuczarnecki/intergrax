# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q3 real web-search qualification orchestrator."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_Q3_PROOF_PACKAGE_DIR = Path(__file__).resolve().parent
for _path in (_REPO_ROOT, _REPO_ROOT / "agents", _REPO_ROOT / "applications"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from scripts.proof.intergrax_proof_environment import load_proof_environment

load_proof_environment(
    proof_package_dir=_Q3_PROOF_PACKAGE_DIR,
    repository_root=_REPO_ROOT,
)

import json
import os
import shutil
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
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessment
from intergrax.runtime.diagnostics.diagnostic_assessment_composer import DiagnosticAssessmentComposer
from intergrax.runtime.diagnostics.functional_diagnostic_analyzer import FunctionalDiagnosticAnalyzer
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import FunctionalDiagnosticCheckStatus
from intergrax.runtime.diagnostics.functional_operator_projection import FunctionalOperatorOutcomeStatus
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
from intergrax.runtime.diagnostics.specifications.q3_web_search_functional_diagnostic_specification import (
    Q3_WEB_SEARCH_OPERATION_ID,
    build_q3_web_search_functional_diagnostic_specification,
)
from intergrax.runtime.observability.functional_evidence_runtime_wiring import (
    wire_in_memory_functional_evidence_runtime,
)
from tests.system.functional_diagnostics_q3.cases import (
    HEALTHY_TASK,
    MANDATORY_CASES,
    Q3_C_WRONG_SOURCE,
    Q3_G_HEALTHY,
    Q3_G_WRONG_SOURCE,
    _REPEAT_CASE_ID,
    case_metadata,
)
from tests.system.functional_diagnostics_q3.oracle import (
    bounded_provider_results,
    build_extraction_validation_evidence,
    build_final_validation_evidence,
    build_query_validation_evidence,
    independent_web_oracle_passes,
    official_source_present_in_candidates,
    resolve_expected_official_source_ref,
)
from tests.system.unified_execution.proof_runner.contracts import ProofConfig
from tests.system.unified_execution.proof_runner.lkw_client import LkwClient, LkwClientError, LkwRunResponse
from web_search_qualifier.search_provider_resolver import (
    preflight_search_provider,
    resolve_qualification_search_provider,
)
from web_search_qualifier.steps.web_search_job import _match_url_from_response
from web_search_qualifier.url_identity import is_expected_python_3120_release_source
from web_search_qualifier.web_search import WebSearchCandidate

_ARTIFACT_DIR = Path(
    os.environ.get(
        "DIAG_FUNCTIONAL_Q3_ARTIFACT_DIR",
        ".tmp/session/diag-functional-q3",
    ),
)
_CURSOR_SECRET = "diag-functional-q3-local-only-secret-32bytes!!"


@dataclass(frozen=True, slots=True)
class DecisionFidelitySnapshot:
    raw_selector_response: str | None
    raw_selector_url: str | None
    raw_extractor_response: str | None
    selection_decision_fidelity_match: bool
    extraction_decision_fidelity_match: bool
    post_decision_forcing_detected: bool


@dataclass(frozen=True, slots=True)
class EvidenceFidelitySnapshot:
    actual_query: str | None
    provider_invoked_with_query: str | None
    query_fidelity_match: bool
    provider_candidate_refs: tuple[str, ...]
    actual_selected_source: str | None
    emitted_selected_source: str | None
    actual_extracted_fact: str | None
    candidate_fidelity_match: bool
    selection_fidelity_match: bool
    extraction_fidelity_match: bool
    validation_fidelity_match: bool
    identity_fidelity_match: bool
    failure_injection_layer: str | None


@dataclass(frozen=True, slots=True)
class QualificationRunRecord:
    case_id: str
    task_id: str
    run_id: str
    provider_id: str | None
    execution_outcome: QualificationExecutionOutcome
    functional_outcome: QualificationFunctionalOutcome
    comparison: QualificationCaseComparison
    evidence_fidelity: EvidenceFidelitySnapshot
    decision_fidelity: DecisionFidelitySnapshot
    diag_first_failed_check: str | None
    operator_outcome: str | None
    actual_query: str | None
    provider_results: tuple[dict[str, str], ...]
    expected_source: str
    actual_selected_source: str | None
    actual_extracted_fact: str | None
    expected_fact: str
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
    provider_id: str | None = None
    selection_decision_fidelity_percent: float = 100.0
    extraction_decision_fidelity_percent: float = 100.0
    post_decision_forcing: str = "NONE"


_EXPECTATION_BY_CASE_ID: dict[str, QualificationCaseExpectation] = {
    case.case_id: case for case in MANDATORY_CASES
}
_EXPECTATION_BY_CASE_ID["Q3-G-A"] = Q3_G_HEALTHY
_EXPECTATION_BY_CASE_ID["Q3-G-B"] = Q3_G_WRONG_SOURCE
_EXPECTATION_BY_CASE_ID[_REPEAT_CASE_ID] = Q3_C_WRONG_SOURCE


def _expectation_for_record(record: QualificationRunRecord) -> QualificationCaseExpectation:
    if record.repeat_group == _REPEAT_CASE_ID:
        return Q3_C_WRONG_SOURCE
    return _EXPECTATION_BY_CASE_ID.get(record.case_id, Q3_C_WRONG_SOURCE)


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
        tenant_id=os.environ.get("DIAG_FUNCTIONAL_Q3_TENANT_ID", "tenant-ue-11g-c1"),
        workspace_id=os.environ.get("DIAG_FUNCTIONAL_Q3_WORKSPACE_ID", "ue-11g-c1-workspace"),
        collection_id=os.environ.get("DIAG_FUNCTIONAL_Q3_COLLECTION_ID", "ue-11g-c1-collection"),
        fixture_root=os.environ.get("DIAG_FUNCTIONAL_Q3_FIXTURE_ROOT", "/cert-fixtures/workspace"),
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


def _query_from_items(items: tuple[PlatformFunctionalEvidence, ...]) -> str | None:
    for item in items:
        if item.kind is not PipelineEvidenceKind.OPERATION_OUTCOME or item.operation_outcome is None:
            continue
        if item.provenance.operation_id != Q3_WEB_SEARCH_OPERATION_ID:
            continue
        return item.operation_outcome.operation_name
    return None


def _extracted_fact_from_items(items: tuple[PlatformFunctionalEvidence, ...]) -> str | None:
    for item in items:
        if item.kind is not PipelineEvidenceKind.OUTPUT_RELATION or item.output_relation is None:
            continue
        ref = item.output_relation.output_artifact_ref.artifact_ref
        if ref.startswith("fact:"):
            return ref[len("fact:") :]
    return None


def _summary_from_response(response: LkwRunResponse) -> dict[str, object]:
    if response.lkw_evidence is None:
        return {}
    diagnostics = response.lkw_evidence.diagnostics
    summary = diagnostics.get("web_search_summary")
    if isinstance(summary, dict):
        return summary
    return {}


def _actual_query_from_response(response: LkwRunResponse, items: tuple[PlatformFunctionalEvidence, ...]) -> str | None:
    summary = _summary_from_response(response)
    actual = summary.get("actual_query")
    if isinstance(actual, str) and actual.strip():
        return actual.strip()
    return _query_from_items(items)


def _selected_source_from_response(
    response: LkwRunResponse,
    items: tuple[PlatformFunctionalEvidence, ...],
) -> str | None:
    summary = _summary_from_response(response)
    selected_ref = summary.get("selected_artifact_ref")
    if isinstance(selected_ref, str) and selected_ref.strip():
        return selected_ref.strip()
    return _selection_ref_from_items(items)


def _extracted_fact_from_response(
    response: LkwRunResponse,
    items: tuple[PlatformFunctionalEvidence, ...],
) -> str | None:
    summary = _summary_from_response(response)
    extracted = summary.get("extracted_fact")
    if isinstance(extracted, str) and extracted.strip():
        return extracted.strip()
    return _extracted_fact_from_items(items)


def _evidence_fidelity_snapshot(
    *,
    actual_query: str | None,
    provider_candidates: tuple[str, ...],
    actual_selected: str | None,
    emitted_selected: str | None,
    actual_extracted: str | None,
    emitted_extracted: str | None,
    remote_items: tuple[PlatformFunctionalEvidence, ...],
    scope: PipelineEvidenceScope,
    failure_injection_layer: str | None,
    validation_expected: bool,
    validation_actual_pass: bool,
) -> EvidenceFidelitySnapshot:
    emitted_candidates = _candidate_refs_from_items(remote_items)
    emitted_query = _query_from_items(remote_items)
    identity_ok = all(
        item.scope.tenant_id == scope.tenant_id
        and item.scope.task_id == scope.task_id
        and item.scope.run_id == scope.run_id
        for item in remote_items
    )
    return EvidenceFidelitySnapshot(
        actual_query=actual_query,
        provider_invoked_with_query=emitted_query,
        query_fidelity_match=(actual_query or "") == (emitted_query or ""),
        provider_candidate_refs=provider_candidates,
        actual_selected_source=actual_selected,
        emitted_selected_source=emitted_selected,
        actual_extracted_fact=actual_extracted,
        candidate_fidelity_match=provider_candidates == emitted_candidates,
        selection_fidelity_match=actual_selected == emitted_selected,
        extraction_fidelity_match=actual_extracted == emitted_extracted,
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
        record.actual_selected_source or "",
        record.actual_query or "",
        record.diag_first_failed_check or "",
        record.operator_outcome or "",
        ",".join(failed_checks),
    )


def _run_case(
    client: LkwClient,
    config: ProofConfig,
    expectation: QualificationCaseExpectation,
    *,
    provider_id: str,
    repeat_group: str | None = None,
) -> QualificationRunRecord:
    metadata = {
        "tenant_id": config.tenant_id,
        **case_metadata(expectation),
    }
    failure_layer_raw = metadata.get("qualification_failure_injection_layer")
    failure_layer = str(failure_layer_raw) if failure_layer_raw is not None else None

    response = client.run_web_search_qualification(
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
    actual_query = _actual_query_from_response(response, remote_items)
    actual_selected = _selected_source_from_response(response, remote_items)
    emitted_selected = _selection_ref_from_items(remote_items)
    actual_extracted = _extracted_fact_from_response(response, remote_items)
    emitted_extracted = _extracted_fact_from_items(remote_items)

    if expectation.case_id in {"Q3-C", "Q3-G-B", _REPEAT_CASE_ID}:
        if not official_source_present_in_candidates(provider_candidates):
            raise LkwClientError(
                f"q3_source_precondition_failed:{expectation.case_id}",
            )

    functional_outcome = (
        QualificationFunctionalOutcome.PASSED
        if independent_web_oracle_passes(
            answer=response.answer,
            actual_query=actual_query,
            selected_source_ref=actual_selected,
            extracted_fact=actual_extracted,
            candidate_refs=provider_candidates,
        )
        else QualificationFunctionalOutcome.FAILED
    )

    pipeline_scope = PipelineEvidenceScope(
        tenant_id=config.tenant_id,
        task_id=validate_task_id(response.task_id),
        run_id=validate_run_id(response.run_id),
    )
    expected_source_ref = resolve_expected_official_source_ref(provider_candidates)
    query_validation = build_query_validation_evidence(
        pipeline_scope,
        actual_query=actual_query,
        idempotency_key=expectation.case_id,
    )
    extraction_validation = build_extraction_validation_evidence(
        pipeline_scope,
        extracted_fact=actual_extracted,
        idempotency_key=expectation.case_id,
    )
    final_validation = build_final_validation_evidence(
        pipeline_scope,
        answer=response.answer,
        extracted_fact=actual_extracted,
        idempotency_key=expectation.case_id,
    )
    spec = build_q3_web_search_functional_diagnostic_specification(
        query_validation_id=query_validation.validation_id,
        extraction_validation_id=extraction_validation.validation_id,
        final_validation_id=(
            final_validation.validation_id if expectation.include_validation else None
        ),
        include_query_validation=True,
        include_extraction_validation=True,
        include_final_validation=expectation.include_validation,
        expected_selection_artifact_ref=expected_source_ref,
    )
    wiring = wire_in_memory_functional_evidence_runtime(cursor_secret=_CURSOR_SECRET)
    for item in remote_items:
        wiring.persistence.append(item)
    validations = [query_validation, extraction_validation]
    if expectation.include_validation:
        validations.append(final_validation)
    validations_lookup = FunctionalValidationEvidenceLookup.for_scope(
        tenant_id=config.tenant_id,
        task_id=pipeline_scope.task_id,
        run_id=pipeline_scope.run_id,
        attempt_id=None,
        validations=tuple(validations),
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
    summary = _summary_from_response(response)
    fidelity = _evidence_fidelity_snapshot(
        actual_query=actual_query,
        provider_candidates=provider_candidates,
        actual_selected=actual_selected,
        emitted_selected=emitted_selected,
        actual_extracted=actual_extracted,
        emitted_extracted=emitted_extracted,
        remote_items=remote_items,
        scope=pipeline_scope,
        failure_injection_layer=failure_layer,
        validation_expected=expectation.include_validation,
        validation_actual_pass=final_validation.outcome.value == "passed",
    )
    decision_fidelity = _decision_fidelity_snapshot(
        summary=summary,
        actual_selected=actual_selected,
        emitted_selected=emitted_selected,
        actual_extracted=actual_extracted,
        emitted_extracted=emitted_extracted,
    )
    return QualificationRunRecord(
        case_id=expectation.case_id,
        task_id=response.task_id,
        run_id=response.run_id,
        provider_id=provider_id,
        execution_outcome=execution_outcome,
        functional_outcome=functional_outcome,
        comparison=comparison,
        evidence_fidelity=fidelity,
        decision_fidelity=decision_fidelity,
        diag_first_failed_check=(
            str(analysis.first_proven_failure) if analysis.first_proven_failure is not None else None
        ),
        operator_outcome=(
            operator.functional_projection.outcome_status.value
            if operator.functional_projection is not None
            else None
        ),
        actual_query=actual_query,
        provider_results=tuple(bounded_provider_results(provider_candidates)),
        expected_source=expected_source_ref,
        actual_selected_source=actual_selected,
        actual_extracted_fact=actual_extracted,
        expected_fact="2023-10-02",
        repeat_group=repeat_group,
    )


def _summary_field(summary: dict[str, object], key: str) -> str | None:
    value = summary.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _candidate_urls_from_summary(summary: dict[str, object]) -> tuple[str, ...]:
    raw = summary.get("candidate_urls")
    if not isinstance(raw, list):
        return ()
    return tuple(str(item) for item in raw if isinstance(item, str))


def _selection_url_from_raw_response(
    raw_response: str | None,
    *,
    candidate_urls: tuple[str, ...],
) -> str | None:
    if not raw_response:
        return None
    candidates = tuple(
        WebSearchCandidate(
            rank=index + 1,
            url=url,
            title=url,
            snippet="",
            provider="tavily",
        )
        for index, url in enumerate(candidate_urls)
    )
    return _match_url_from_response(raw_response, candidates)


def _decision_fidelity_snapshot(
    *,
    summary: dict[str, object],
    actual_selected: str | None,
    emitted_selected: str | None,
    actual_extracted: str | None,
    emitted_extracted: str | None,
) -> DecisionFidelitySnapshot:
    raw_selector_response = _summary_field(summary, "raw_selector_response")
    raw_extractor_response = _summary_field(summary, "raw_extractor_response")
    candidate_urls = _candidate_urls_from_summary(summary)
    raw_selector_url = _selection_url_from_raw_response(
        raw_selector_response,
        candidate_urls=candidate_urls,
    )
    selection_decision_match = True
    if raw_selector_url is not None:
        selection_decision_match = (
            raw_selector_url == (actual_selected or "")
            and raw_selector_url == (emitted_selected or "")
        )
    elif raw_selector_response and actual_selected:
        selection_decision_match = raw_selector_response.strip() == actual_selected.strip()

    extraction_decision_match = True
    if raw_extractor_response is not None and actual_extracted is not None:
        extraction_decision_match = (
            raw_extractor_response.strip() == actual_extracted.strip()
            and raw_extractor_response.strip() == (emitted_extracted or "").strip()
        )

    post_decision_forcing = False
    if raw_selector_url is not None and actual_selected is not None:
        if (
            is_expected_python_3120_release_source(raw_selector_url)
            and not is_expected_python_3120_release_source(actual_selected)
        ):
            post_decision_forcing = True
    if raw_extractor_response and actual_extracted:
        lowered_raw = raw_extractor_response.lower()
        if any(
            marker in lowered_raw
            for marker in ("2023-10-02", "october 2, 2023", "oct. 2, 2023")
        ) and "2023-10-01" in actual_extracted:
            post_decision_forcing = True

    return DecisionFidelitySnapshot(
        raw_selector_response=raw_selector_response,
        raw_selector_url=raw_selector_url,
        raw_extractor_response=raw_extractor_response,
        selection_decision_fidelity_match=selection_decision_match,
        extraction_decision_fidelity_match=extraction_decision_match,
        post_decision_forcing_detected=post_decision_forcing,
    )


def _post_decision_forcing_static_gate() -> bool:
    import ast

    job_path = (
        Path(__file__).resolve().parents[3]
        / "agents"
        / "web_search_qualifier"
        / "steps"
        / "web_search_job.py"
    )
    tree = ast.parse(job_path.read_text(encoding="utf-8"))
    forbidden = (
        "intergrax.runtime.diagnostics",
        "functional_diagnostic",
        "functional_diagnostics_q3",
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
    if "is_expected_python_3120_release_source(selected)" in source:
        return False
    if "_looks_like_correct_python_3120_release_date" in source:
        return False
    return True


def _decision_diagnostics_independence_gate() -> bool:
    return _post_decision_forcing_static_gate()


def _preflight(config: ProofConfig) -> tuple[LkwClient, str]:
    try:
        resolved = resolve_qualification_search_provider()
        preflight_search_provider(resolved)
    except IntegrationConfigurationError as exc:
        raise LkwClientError(f"search_provider_blocked:{exc}") from exc
    client = LkwClient(config)
    client.wait_until_ready()
    return client, resolved.provider_id


def run_qualification() -> QualificationReport:
    config = _config_from_env()
    try:
        client, provider_id = _preflight(config)
    except LkwClientError as exc:
        return _blocked_report(str(exc))

    records: list[QualificationRunRecord] = []
    try:
        for case in MANDATORY_CASES:
            records.append(_run_case(client, config, case, provider_id=provider_id))

        records.append(
            _run_case(client, config, Q3_G_HEALTHY, provider_id=provider_id, repeat_group="isolation"),
        )
        records.append(
            _run_case(
                client,
                config,
                Q3_G_WRONG_SOURCE,
                provider_id=provider_id,
                repeat_group="isolation",
            ),
        )

        repeat_records: list[QualificationRunRecord] = []
        for _ in range(3):
            repeat_records.append(
                _run_case(
                    client,
                    config,
                    Q3_C_WRONG_SOURCE,
                    provider_id=provider_id,
                    repeat_group=_REPEAT_CASE_ID,
                ),
            )
        records.extend(repeat_records)
    except LkwClientError as exc:
        return _blocked_report(str(exc), provider_id=provider_id, partial_records=records)

    wrong_source_records = [
        record
        for record in records
        if record.case_id in {"Q3-C", "Q3-G-B"} or record.repeat_group == _REPEAT_CASE_ID
    ]
    for record in wrong_source_records:
        if is_expected_python_3120_release_source(record.actual_selected_source or ""):
            return _blocked_report(
                "q3_c_not_inducible:llm_selected_canonical_despite_bias",
                provider_id=provider_id,
                partial_records=records,
            )

    q3_d_records = [record for record in records if record.case_id == "Q3-D"]
    for record in q3_d_records:
        extracted = (record.actual_extracted_fact or "").lower()
        if any(
            marker in extracted
            for marker in ("2023-10-02", "october 2, 2023", "oct. 2, 2023")
        ):
            return _blocked_report(
                "q3_d_not_inducible:llm_extracted_correct_date_despite_bias",
                provider_id=provider_id,
                partial_records=records,
            )

    repeatability_pass = len({_semantic_signature(item) for item in repeat_records}) == 1
    fidelity_pass = all(
        record.evidence_fidelity.candidate_fidelity_match
        and record.evidence_fidelity.selection_fidelity_match
        and record.evidence_fidelity.query_fidelity_match
        and record.evidence_fidelity.extraction_fidelity_match
        and record.evidence_fidelity.identity_fidelity_match
        for record in records
        if record.case_id != "Q3-F"
    )
    decision_independence_pass = _decision_diagnostics_independence_gate()
    selection_decision_fidelity_pass = all(
        record.decision_fidelity.selection_decision_fidelity_match for record in records
    )
    extraction_decision_fidelity_pass = all(
        record.decision_fidelity.extraction_decision_fidelity_match
        for record in records
        if record.actual_extracted_fact
    )
    post_decision_forcing_pass = all(
        not record.decision_fidelity.post_decision_forcing_detected for record in records
    )
    selection_decision_fidelity_percent = (
        sum(1 for record in records if record.decision_fidelity.selection_decision_fidelity_match)
        / len(records)
        * 100.0
        if records
        else 0.0
    )
    extraction_records = [record for record in records if record.actual_extracted_fact]
    extraction_decision_fidelity_percent = (
        sum(
            1
            for record in extraction_records
            if record.decision_fidelity.extraction_decision_fidelity_match
        )
        / len(extraction_records)
        * 100.0
        if extraction_records
        else 100.0
    )
    post_decision_forcing = "NONE" if post_decision_forcing_pass else "DETECTED"

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
        if item.case_id in {"Q3-A", "Q3-G-A"}
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
        if item.case_id == "Q3-F" and item.comparison.result is QualificationComparisonResult.MATCH
    )
    stage_accuracy = (stage_matched / len(records) * 100.0) if records else 0.0
    inconclusive_accuracy = 100.0 if inconclusive_correct == 1 else 0.0

    verdict = (
        "PASS"
        if mismatched == 0
        and repeatability_pass
        and fidelity_pass
        and decision_independence_pass
        and selection_decision_fidelity_pass
        and extraction_decision_fidelity_pass
        and post_decision_forcing_pass
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
        provider_id=provider_id,
        selection_decision_fidelity_percent=selection_decision_fidelity_percent,
        extraction_decision_fidelity_percent=extraction_decision_fidelity_percent,
        post_decision_forcing=post_decision_forcing,
    )
    _write_artifact(
        report,
        fidelity_pass=fidelity_pass,
        decision_diagnostics_independence_pass=decision_independence_pass,
        selection_decision_fidelity_pass=selection_decision_fidelity_pass,
        extraction_decision_fidelity_pass=extraction_decision_fidelity_pass,
    )
    return report


def _blocked_report(
    reason: str,
    *,
    provider_id: str | None = None,
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
        provider_id=provider_id,
    )
    _write_artifact(report, fidelity_pass=False, decision_diagnostics_independence_pass=False)
    return report


def _write_artifact(
    report: QualificationReport,
    *,
    fidelity_pass: bool,
    decision_diagnostics_independence_pass: bool,
    selection_decision_fidelity_pass: bool = False,
    extraction_decision_fidelity_pass: bool = False,
) -> None:
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _ARTIFACT_DIR / "qualification-report.json"
    initial_path = _ARTIFACT_DIR / "qualification-report-initial-failure.json"
    if report_path.exists() and not initial_path.exists():
        shutil.copy2(report_path, initial_path)
    payload = {
        "verdict": report.verdict,
        "blocked_reason": report.blocked_reason,
        "provider_id": report.provider_id,
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
        "decision_diagnostics_independence_pass": decision_diagnostics_independence_pass,
        "selection_decision_fidelity_pass": selection_decision_fidelity_pass,
        "extraction_decision_fidelity_pass": extraction_decision_fidelity_pass,
        "selection_decision_fidelity_percent": report.selection_decision_fidelity_percent,
        "extraction_decision_fidelity_percent": report.extraction_decision_fidelity_percent,
        "post_decision_forcing": report.post_decision_forcing,
        "records": [
            {
                "case_id": record.case_id,
                "task_id": record.task_id,
                "run_id": record.run_id,
                "provider": record.provider_id,
                "actual_query": record.actual_query,
                "provider_results": list(record.provider_results),
                "expected_source": record.expected_source,
                "actual_selected_source": record.actual_selected_source,
                "actual_extracted_fact": record.actual_extracted_fact,
                "expected_fact": record.expected_fact,
                "functional_ground_truth": record.functional_outcome.value,
                "expected_first_failure": (
                    str(_expectation_for_record(record).expected_first_proven_failed_check)
                    if _expectation_for_record(record).expected_first_proven_failed_check is not None
                    else None
                ),
                "actual_first_failure": record.diag_first_failed_check,
                "operator_outcome": record.operator_outcome,
                "comparison_result": record.comparison.result.value,
                "repeat_group": record.repeat_group,
                "evidence_fidelity": {
                    "query_fidelity_match": record.evidence_fidelity.query_fidelity_match,
                    "candidate_fidelity_match": record.evidence_fidelity.candidate_fidelity_match,
                    "selection_fidelity_match": record.evidence_fidelity.selection_fidelity_match,
                    "extraction_fidelity_match": record.evidence_fidelity.extraction_fidelity_match,
                    "validation_fidelity_match": record.evidence_fidelity.validation_fidelity_match,
                    "identity_fidelity_match": record.evidence_fidelity.identity_fidelity_match,
                },
                "decision_fidelity": {
                    "raw_selector_response": record.decision_fidelity.raw_selector_response,
                    "raw_selector_url": record.decision_fidelity.raw_selector_url,
                    "raw_extractor_response": record.decision_fidelity.raw_extractor_response,
                    "selection_decision_fidelity_match": (
                        record.decision_fidelity.selection_decision_fidelity_match
                    ),
                    "extraction_decision_fidelity_match": (
                        record.decision_fidelity.extraction_decision_fidelity_match
                    ),
                    "post_decision_forcing_detected": (
                        record.decision_fidelity.post_decision_forcing_detected
                    ),
                },
            }
            for record in report.records
        ],
    }
    (report_path).write_text(
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
