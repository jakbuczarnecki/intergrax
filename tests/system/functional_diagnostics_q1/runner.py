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
    HEALTHY_QUERY,
    MANDATORY_CASES,
    Q1_A_HEALTHY,
    Q1_B_SELECTION_FAILURE,
    Q1_C_SYNTHESIS_FAILURE,
    Q1_E_FAILURE,
    Q1_E_HEALTHY,
    Q1_H_HISTORICAL_WRONG_DATE,
    case_metadata,
)
from tests.system.functional_diagnostics_q1.oracle import (
    build_independent_validation_evidence,
    evidence_texts_from_lkw_response,
    independent_date_oracle_passes,
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
class EvidenceFidelitySnapshot:
    provider_candidate_refs: tuple[str, ...]
    actual_selected_artifact: str | None
    emitted_selected_artifact: str | None
    candidate_fidelity_match: bool
    selection_fidelity_match: bool
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
    repeat_group: str | None = None
    historical_reproduction: str | None = None


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
    expected_selection_artifact_ref: str | None
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


def _fixture_text_by_leaf(config: ProofConfig) -> dict[str, str]:
    root = config.fixture_root.rstrip("/\\")
    if root == _DOCKER_FIXTURE_ROOT:
        local_root = Path(__file__).resolve().parents[1] / "unified_execution" / "fixtures" / "workspace"
    else:
        local_root = Path(root)
    texts: dict[str, str] = {}
    for name in _FIXTURE_FILES:
        path = local_root / name
        if path.is_file():
            texts[name] = path.read_text(encoding="utf-8")
    return texts


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


def _provider_candidates_from_search_response(
    response: LkwRunResponse,
    config: ProofConfig,
) -> tuple[str, ...]:
    remote = _fetch_functional_evidence(
        config,
        tenant_id=config.tenant_id,
        task_id=response.task_id,
        run_id=response.run_id,
    )
    return _candidate_refs_from_items(remote)


def _actual_selected_from_search_response(
    response: LkwRunResponse,
    config: ProofConfig,
) -> str | None:
    remote = _fetch_functional_evidence(
        config,
        tenant_id=config.tenant_id,
        task_id=response.task_id,
        run_id=response.run_id,
    )
    return _selection_ref_from_items(remote)


def _build_synthesis_handoff_evidence(
    *,
    candidate_refs: tuple[str, ...],
    fixture_texts: dict[str, str],
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for ref in candidate_refs:
        if ref.startswith("source:"):
            leaf = ref.removeprefix("source:")
            text = fixture_texts.get(leaf, "")
        elif ref.startswith("chunk:"):
            text = fixture_texts.get("incident-report.md", "")
        else:
            text = ""
        item: dict[str, object] = {"text": text}
        if ref.startswith("chunk:"):
            item["chunk_id"] = ref.removeprefix("chunk:")
        if ref.startswith("source:"):
            item["source_path"] = ref.removeprefix("source:")
        evidence.append(item)
    return evidence


def _evidence_fidelity_snapshot(
    *,
    provider_candidates: tuple[str, ...],
    actual_selected: str | None,
    emitted_selected: str | None,
    scope: PipelineEvidenceScope,
    remote_items: tuple[PlatformFunctionalEvidence, ...],
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
        actual_selected_artifact=actual_selected,
        emitted_selected_artifact=emitted_selected,
        candidate_fidelity_match=provider_candidates == emitted_candidates,
        selection_fidelity_match=actual_selected == emitted_selected,
        validation_fidelity_match=(
            validation_expected == validation_actual_pass
            if validation_expected
            else True
        ),
        identity_fidelity_match=identity_ok,
        failure_injection_layer=failure_injection_layer,
    )


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


def _discover_expected_selection_artifact(
    client: LkwClient,
    config: ProofConfig,
) -> str:
    response = client.run_search(
        message=HEALTHY_QUERY,
        metadata={"query": HEALTHY_QUERY},
    )
    if response.state != "completed":
        raise LkwClientError(f"discovery_search_state_{response.state}")
    selected = _actual_selected_from_search_response(response, config)
    if selected is None:
        raise LkwClientError("discovery_selection_missing")
    return selected


def _run_synthesis_case(
    client: LkwClient,
    config: ProofConfig,
    metadata: dict[str, object],
    *,
    fixture_texts: dict[str, str],
) -> tuple[LkwRunResponse, LkwRunResponse, tuple[PlatformFunctionalEvidence, ...], str | None]:
    search_response = client.run_search(
        message=str(metadata.get("query") or search_request_message()),
        metadata=metadata,
    )
    search_remote = _fetch_functional_evidence(
        config,
        tenant_id=config.tenant_id,
        task_id=search_response.task_id,
        run_id=search_response.run_id,
    )
    candidate_refs = _candidate_refs_from_items(search_remote)
    selected_ref = _selection_ref_from_items(search_remote)
    handoff_evidence = _build_synthesis_handoff_evidence(
        candidate_refs=candidate_refs,
        fixture_texts=fixture_texts,
    )
    synth_metadata = {
        **metadata,
        "shadow_workspace": True,
        "output_name": metadata.get("output_name", "q1-synthesis-draft.md"),
        "evidence": handoff_evidence,
        "selected_artifact_ref": selected_ref,
        "search_summary": {
            "query": metadata.get("query", search_request_message()),
            "num_results": len(handoff_evidence),
            "evidence": handoff_evidence,
            "selected_artifact_ref": selected_ref,
        },
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
    draft_text = metadata.get("draft")
    synthesis_draft = draft_text.strip() if isinstance(draft_text, str) and draft_text.strip() else None
    return search_response, synth_response, merged, synthesis_draft


def _run_case(
    client: LkwClient,
    config: ProofConfig,
    expectation: QualificationCaseExpectation,
    *,
    expected_selection_artifact_ref: str,
    fixture_texts: dict[str, str],
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
    failure_layer_raw = metadata.get("qualification_failure_injection_layer")
    failure_layer = str(failure_layer_raw) if failure_layer_raw is not None else None

    remote_items: tuple[PlatformFunctionalEvidence, ...] | None = None
    synthesis_draft: str | None = None
    scope_response: LkwRunResponse
    if expectation.include_output_relation:
        scope_response, response, remote_items, synthesis_draft = _run_synthesis_case(
            client,
            config,
            metadata,
            fixture_texts=fixture_texts,
        )
    else:
        response = client.run_search(
            message=str(metadata.get("query") or search_request_message()),
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
        synthesis_draft_text=synthesis_draft,
    )
    functional_outcome = (
        QualificationFunctionalOutcome.PASSED
        if independent_date_oracle_passes(answer=response.answer, evidence_texts=evidence_texts)
        else QualificationFunctionalOutcome.FAILED
    )

    if remote_items is None:
        remote_items = _fetch_functional_evidence(
            config,
            tenant_id=config.tenant_id,
            task_id=response.task_id,
            run_id=response.run_id,
        )

    provider_candidates = _provider_candidates_from_search_response(scope_response, config)
    actual_selected = _actual_selected_from_search_response(scope_response, config)
    emitted_selected = _selection_ref_from_items(remote_items)

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
    )
    spec = build_c1_rag_functional_diagnostic_specification(
        validation_id=validation.validation_id if expectation.include_validation else None,
        include_output_relation=expectation.include_output_relation,
        include_validation=expectation.include_validation,
        expected_selection_artifact_ref=expected_selection_artifact_ref,
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

    historical_reproduction: str | None = None
    if expectation.case_id == "Q1-H":
        historical_reproduction = (
            "NOT_REPRODUCED"
            if functional_outcome is QualificationFunctionalOutcome.PASSED
            else "REPRODUCED"
        )

    fidelity = _evidence_fidelity_snapshot(
        provider_candidates=provider_candidates,
        actual_selected=actual_selected,
        emitted_selected=emitted_selected,
        scope=pipeline_scope,
        remote_items=remote_items,
        failure_injection_layer=failure_layer,
        validation_expected=expectation.include_validation,
        validation_actual_pass=independent_date_oracle_passes(
            answer=response.answer,
            evidence_texts=evidence_texts,
        ),
    )

    return QualificationRunRecord(
        case_id=expectation.case_id,
        task_id=scope_response.task_id,
        run_id=scope_response.run_id,
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
        repeat_group=repeat_group,
        historical_reproduction=historical_reproduction,
    )


def _decision_diagnostics_independence_gate() -> bool:
    """Static gate: search selection policy has zero DIAG imports."""
    import ast

    selection_path = Path(__file__).resolve().parents[3] / "agents" / "local_search" / "retrieval_selection.py"
    tree = ast.parse(selection_path.read_text(encoding="utf-8"))
    forbidden = (
        "intergrax.runtime.diagnostics",
        "functional_diagnostic",
        "functional_diagnostics_q1",
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


def run_evidence_independence_probe(
    client: LkwClient,
    config: ProofConfig,
    *,
    expected_selection_artifact_ref: str,
) -> bool:
    """Same real run, different DIAG spec expectation — emitted evidence must match."""
    metadata = {
        "tenant_id": config.tenant_id,
        "workspace_id": config.workspace_id,
        "collection_id": config.collection_id,
        "query": HEALTHY_QUERY,
        "top_k": 5,
        **case_metadata(Q1_A_HEALTHY),
    }
    response = client.run_search(message=HEALTHY_QUERY, metadata=metadata)
    first_items = _fetch_functional_evidence(
        config,
        tenant_id=config.tenant_id,
        task_id=response.task_id,
        run_id=response.run_id,
    )
    first_selection = _selection_ref_from_items(first_items)

    alt_spec = build_c1_rag_functional_diagnostic_specification(
        expected_selection_artifact_ref="chunk:definitely-not-the-real-selection",
        include_validation=False,
    )
    wiring = wire_in_memory_functional_evidence_runtime(cursor_secret=_CURSOR_SECRET)
    _replay_into_persistence(wiring.persistence, first_items)
    scope = PipelineEvidenceScope(
        tenant_id=config.tenant_id,
        task_id=validate_task_id(response.task_id),
        run_id=validate_run_id(response.run_id),
    )
    FunctionalDiagnosticAnalyzer(wiring.persistence).analyze(
        tenant_id=config.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=alt_spec,
        validations=None,
    )

    second_items = _fetch_functional_evidence(
        config,
        tenant_id=config.tenant_id,
        task_id=response.task_id,
        run_id=response.run_id,
    )
    second_selection = _selection_ref_from_items(second_items)
    return first_selection == second_selection


def run_qualification() -> QualificationReport:
    config = _config_from_env()
    client = LkwClient(config)
    fixture_texts = _fixture_text_by_leaf(config)
    try:
        client.wait_until_ready()
    except LkwClientError as exc:
        return _blocked_report(str(exc))

    index_response = client.run_index(source_paths=_fixture_paths(config))
    if index_response.state != "completed":
        return _blocked_report(f"index_state_{index_response.state}")

    try:
        expected_selection = _discover_expected_selection_artifact(client, config)
    except LkwClientError as exc:
        return _blocked_report(str(exc))

    records: list[QualificationRunRecord] = []
    for case in MANDATORY_CASES:
        records.append(
            _run_case(
                client,
                config,
                case,
                expected_selection_artifact_ref=expected_selection,
                fixture_texts=fixture_texts,
            ),
        )

    records.append(
        _run_case(
            client,
            config,
            Q1_E_HEALTHY,
            expected_selection_artifact_ref=expected_selection,
            fixture_texts=fixture_texts,
            repeat_group="isolation",
        ),
    )
    records.append(
        _run_case(
            client,
            config,
            Q1_E_FAILURE,
            expected_selection_artifact_ref=expected_selection,
            fixture_texts=fixture_texts,
            repeat_group="isolation",
        ),
    )

    repeat_records: list[QualificationRunRecord] = []
    for _ in range(3):
        repeat_records.append(
            _run_case(
                client,
                config,
                Q1_B_SELECTION_FAILURE,
                expected_selection_artifact_ref=expected_selection,
                fixture_texts=fixture_texts,
                repeat_group=_REPEAT_CASE_ID,
            ),
        )
    records.extend(repeat_records)

    synthesis_repeat_records: list[QualificationRunRecord] = []
    for _ in range(3):
        synthesis_repeat_records.append(
            _run_case(
                client,
                config,
                Q1_C_SYNTHESIS_FAILURE,
                expected_selection_artifact_ref=expected_selection,
                fixture_texts=fixture_texts,
                repeat_group="Q1-C-R",
            ),
        )
    records.extend(synthesis_repeat_records)

    records.append(
        _run_case(
            client,
            config,
            Q1_H_HISTORICAL_WRONG_DATE,
            expected_selection_artifact_ref=expected_selection,
            fixture_texts=fixture_texts,
            repeat_group="historical",
        ),
    )

    repeatability_pass = len({_semantic_signature(item) for item in repeat_records}) == 1
    synthesis_repeatability_pass = len({_semantic_signature(item) for item in synthesis_repeat_records}) == 1
    fidelity_pass = all(
        record.evidence_fidelity.candidate_fidelity_match
        and record.evidence_fidelity.selection_fidelity_match
        and record.evidence_fidelity.identity_fidelity_match
        for record in records
        if record.case_id != "Q1-D"
    )
    independence_pass = run_evidence_independence_probe(
        client,
        config,
        expected_selection_artifact_ref=expected_selection,
    )
    decision_independence_pass = _decision_diagnostics_independence_gate()

    matched = sum(1 for item in records if item.comparison.result is QualificationComparisonResult.MATCH)
    mismatched = len(records) - matched
    false_positives = sum(
        1
        for item in records
        if item.case_id in {"Q1-A", "Q1-E-A", "Q1-H"}
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
    verdict = (
        "PASS"
        if all_matched
        and repeatability_pass
        and synthesis_repeatability_pass
        and fidelity_pass
        and independence_pass
        and decision_independence_pass
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
        repeatability_pass=repeatability_pass and synthesis_repeatability_pass,
        expected_selection_artifact_ref=expected_selection,
        records=tuple(records),
    )
    _write_artifact(
        report,
        independence_pass=independence_pass,
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
        repeatability_pass=False,
        expected_selection_artifact_ref=None,
        records=(),
        blocked_reason=reason,
    )
    _write_artifact(report, independence_pass=False, fidelity_pass=False, decision_diagnostics_independence_pass=False)
    return report


def _write_artifact(
    report: QualificationReport,
    *,
    independence_pass: bool,
    fidelity_pass: bool,
    decision_diagnostics_independence_pass: bool,
) -> None:
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
        "expected_selection_artifact_ref": report.expected_selection_artifact_ref,
        "evidence_independence_pass": independence_pass,
        "evidence_fidelity_pass": fidelity_pass,
        "decision_diagnostics_independence": (
            "PASS" if decision_diagnostics_independence_pass else "FAIL"
        ),
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
                "repeat_group": item.repeat_group,
                "historical_reproduction": item.historical_reproduction,
                "provider_candidates": list(item.evidence_fidelity.provider_candidate_refs),
                "actual_selected_artifact": item.evidence_fidelity.actual_selected_artifact,
                "emitted_selected_artifact": item.evidence_fidelity.emitted_selected_artifact,
                "evidence_fidelity_match": (
                    item.evidence_fidelity.candidate_fidelity_match
                    and item.evidence_fidelity.selection_fidelity_match
                    and item.evidence_fidelity.identity_fidelity_match
                ),
                "failure_injection_layer": item.evidence_fidelity.failure_injection_layer,
                "candidate_fidelity": item.evidence_fidelity.candidate_fidelity_match,
                "selection_fidelity": item.evidence_fidelity.selection_fidelity_match,
                "validation_fidelity": item.evidence_fidelity.validation_fidelity_match,
                "identity_fidelity": item.evidence_fidelity.identity_fidelity_match,
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
