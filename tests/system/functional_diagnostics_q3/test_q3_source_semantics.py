# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q3-R1 source identity, injection, and missing-evidence semantics."""

from __future__ import annotations

import pytest

from datetime import datetime, timezone

from intergrax.contracts.execution_identity import mint_event_id, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.functional_diagnostic_analyzer import FunctionalDiagnosticAnalyzer
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import FunctionalDiagnosticCheckStatus
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineCandidateFact,
    PipelineEvidenceKind,
    PipelineEvidenceProvenance,
    PipelineEvidenceScope,
    PipelineOperationOutcomeFact,
    PipelineOperationStatus,
    PipelineOutputRelationFact,
    PlatformFunctionalEvidence,
)
from intergrax.runtime.observability.export_attributes import ObservabilityArtifactReference
from intergrax.runtime.diagnostics.functional_validation_lookup import FunctionalValidationEvidenceLookup
from intergrax.runtime.diagnostics.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from intergrax.runtime.diagnostics.specifications.q3_web_search_functional_diagnostic_specification import (
    CHECK_Q3_EXTRACTION_VALIDATION,
    CHECK_Q3_SELECTION,
    Q3_WEB_EXTRACT_OPERATION_ID,
    Q3_WEB_QUERY_ID,
    Q3_WEB_SEARCH_OPERATION_ID,
    build_q3_web_search_functional_diagnostic_specification,
)
from tests.system.functional_diagnostics_q3.oracle import (
    CANONICAL_EXPECTED_SOURCE_REF,
    build_extraction_validation_evidence,
    build_query_validation_evidence,
    resolve_expected_official_source_ref,
)
from web_search_qualifier.url_identity import (
    artifact_ref_for_url as agent_artifact_ref_for_url,
    is_expected_python_3120_release_source,
)
from web_search_qualifier.steps.web_search_job import _extract_fact, _select_source
from web_search_qualifier.web_search import WebSearchCandidate

pytestmark = pytest.mark.unit

_RC3_URL = "https://www.python.org/downloads/release/python-3120rc3"
_FINAL_URL = "https://www.python.org/downloads/release/python-3120/"
_BASE_TIME = datetime(2026, 9, 2, 8, 0, tzinfo=timezone.utc)


def _artifact(ref: str) -> ObservabilityArtifactReference:
    return ObservabilityArtifactReference(artifact_ref=ref)


class _StubAdapter:
    def __init__(self, *, selection: str | None = None, extraction: str = "2023-10-02") -> None:
        self._selection = selection
        self._extraction = extraction

    def generate_messages(self, messages, *, temperature: float, run_id: str) -> object:
        del temperature, run_id
        system = messages[0].content
        if "select" in system.lower() or "source url" in system.lower():
            return type("R", (), {"content": self._selection or _FINAL_URL})()
        return type("R", (), {"content": self._extraction})()


def _candidates() -> tuple[WebSearchCandidate, ...]:
    return (
        WebSearchCandidate(
            rank=1,
            url=_RC3_URL,
            title="Python 3.12.0rc3",
            snippet="release candidate",
            provider="tavily",
        ),
        WebSearchCandidate(
            rank=2,
            url=_FINAL_URL,
            title="Python 3.12.0",
            snippet="Released Oct. 2, 2023",
            provider="tavily",
        ),
    )


def test_resolve_expected_source_ignores_candidate_ranking() -> None:
    refs = (
        agent_artifact_ref_for_url(_RC3_URL),
        agent_artifact_ref_for_url(_FINAL_URL),
    )
    assert resolve_expected_official_source_ref(refs) == CANONICAL_EXPECTED_SOURCE_REF


def test_exact_canonical_predicate_distinguishes_rc3_from_final_release() -> None:
    assert is_expected_python_3120_release_source(_FINAL_URL)
    assert not is_expected_python_3120_release_source(_RC3_URL)


def test_healthy_selection_accepts_final_release() -> None:
    selected = _select_source(
        adapter=_StubAdapter(selection=_FINAL_URL),
        run_id="run-test",
        task_message="When was Python 3.12.0 released?",
        candidates=_candidates(),
        failure_layer=None,
    )
    assert is_expected_python_3120_release_source(selected or "")


def test_wrong_source_injection_selects_non_canonical_candidate() -> None:
    selected = _select_source(
        adapter=_StubAdapter(selection=_FINAL_URL),
        run_id="run-test",
        task_message="When was Python 3.12.0 released?",
        candidates=_candidates(),
        failure_layer="source_selection_bias",
    )
    assert selected == _RC3_URL
    assert not is_expected_python_3120_release_source(selected or "")


def test_extraction_injection_replaces_correct_fact_with_wrong_date() -> None:
    extracted = _extract_fact(
        adapter=_StubAdapter(extraction="2023-10-02"),
        run_id="run-test",
        selected_url=_FINAL_URL,
        snippet="Released Oct. 2, 2023",
        failure_layer="extraction_bias",
    )
    assert extracted == "2023-10-01"
    assert extracted != "2023-10-02"


def test_missing_selection_yields_inconclusive_without_proven_extraction_failure() -> None:
    scope = PipelineEvidenceScope(
        tenant_id="tenant-q3",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    query_validation = build_query_validation_evidence(
        scope,
        actual_query="python 3.12.0 release",
        idempotency_key="Q3-F",
    )
    extraction_validation = build_extraction_validation_evidence(
        scope,
        extracted_fact="Oct. 2, 2023",
        idempotency_key="Q3-F",
    )
    spec = build_q3_web_search_functional_diagnostic_specification(
        query_validation_id=query_validation.validation_id,
        extraction_validation_id=extraction_validation.validation_id,
        final_validation_id=None,
        include_final_validation=False,
        expected_selection_artifact_ref=CANONICAL_EXPECTED_SOURCE_REF,
    )
    persistence = InMemoryFunctionalEvidencePersistence()
    persistence.append(
        PlatformFunctionalEvidence(
            evidence_id=mint_event_id(),
            kind=PipelineEvidenceKind.OPERATION_OUTCOME,
            scope=scope,
            provenance=PipelineEvidenceProvenance(
                producer_component="diag.test",
                operation_id=Q3_WEB_SEARCH_OPERATION_ID,
                recorded_at=_BASE_TIME,
            ),
            operation_outcome=PipelineOperationOutcomeFact(
                operation_name=Q3_WEB_SEARCH_OPERATION_ID,
                status=PipelineOperationStatus.SUCCEEDED,
            ),
        ),
    )
    persistence.append(
        PlatformFunctionalEvidence(
            evidence_id=mint_event_id(),
            kind=PipelineEvidenceKind.CANDIDATE_RANK,
            scope=scope,
            provenance=PipelineEvidenceProvenance(
                producer_component="diag.test",
                operation_id=Q3_WEB_SEARCH_OPERATION_ID,
                recorded_at=_BASE_TIME,
            ),
            candidate=PipelineCandidateFact(
                query_id=Q3_WEB_QUERY_ID,
                candidate_artifact_ref=_artifact(CANONICAL_EXPECTED_SOURCE_REF),
                rank=1,
                selected=False,
            ),
        ),
    )
    persistence.append(
        PlatformFunctionalEvidence(
            evidence_id=mint_event_id(),
            kind=PipelineEvidenceKind.OUTPUT_RELATION,
            scope=scope,
            provenance=PipelineEvidenceProvenance(
                producer_component="diag.test",
                operation_id=Q3_WEB_EXTRACT_OPERATION_ID,
                recorded_at=_BASE_TIME,
            ),
            output_relation=PipelineOutputRelationFact(
                selected_artifact_ref=_artifact(CANONICAL_EXPECTED_SOURCE_REF),
                output_artifact_ref=_artifact("fact:Oct. 2, 2023"),
                relation_kind="extracted_from",
            ),
        ),
    )
    analysis = FunctionalDiagnosticAnalyzer(persistence).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=spec,
        validations=FunctionalValidationEvidenceLookup.for_scope(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=None,
            validations=(query_validation, extraction_validation),
        ),
    )
    by_id = {item.check_id: item.status for item in analysis.check_results}
    assert by_id[CHECK_Q3_SELECTION] is FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE
    assert by_id[CHECK_Q3_EXTRACTION_VALIDATION] is FunctionalDiagnosticCheckStatus.PROVEN_PASS
