# © Artur Czarnecki. All rights reserved.

"""Independent deterministic oracle for Q3 web-search qualification."""

from __future__ import annotations

from web_search_qualifier.url_identity import (
    artifact_ref_for_url,
    is_expected_python_3120_release_source,
    is_official_python_release_source,
    url_from_artifact_ref,
)
from intergrax.runtime.diagnostics.functional_evidence import PipelineEvidenceScope
from intergrax.runtime.diagnostics.functional_validation import (
    DiagnosticExecutionCorrelation,
    ExpectedActualRelation,
    FunctionalValidationEvidence,
    FunctionalValidationKind,
    FunctionalValidationOutcome,
    FunctionalValidatorRef,
    functional_validation_evidence_id,
)

Q3_ORACLE_VALIDATOR_ID = "q3.web.functional_oracle.v1"
Q3_QUERY_ORACLE_VALIDATOR_ID = "q3.web.query_oracle.v1"
Q3_EXTRACTION_ORACLE_VALIDATOR_ID = "q3.web.extraction_oracle.v1"

EXPECTED_RELEASE_DATE_PHRASES: tuple[str, ...] = (
    "2023-10-02",
    "October 2, 2023",
    "2 October 2023",
    "Oct 2, 2023",
    "Oct. 2, 2023",
)
CANONICAL_OFFICIAL_SOURCE_URL = "https://www.python.org/downloads/release/python-3120/"
CANONICAL_EXPECTED_SOURCE_REF = artifact_ref_for_url(CANONICAL_OFFICIAL_SOURCE_URL)

HEALTHY_TASK = (
    "When was Python 3.12.0 released according to the official Python website? "
    "Report the release date."
)


def query_intent_matches(actual_query: str) -> bool:
    lowered = actual_query.lower()
    return "python" in lowered and "3.12" in lowered


def official_source_present_in_candidates(candidate_refs: tuple[str, ...]) -> bool:
    for ref in candidate_refs:
        url = url_from_artifact_ref(ref)
        if is_expected_python_3120_release_source(url):
            return True
    return False


def resolve_expected_official_source_ref(candidate_refs: tuple[str, ...]) -> str:
    del candidate_refs
    return CANONICAL_EXPECTED_SOURCE_REF


def extracted_fact_matches_oracle(extracted_fact: str | None) -> bool:
    if not extracted_fact:
        return False
    combined = extracted_fact.lower()
    return any(phrase.lower() in combined for phrase in EXPECTED_RELEASE_DATE_PHRASES)


def final_answer_matches_oracle(answer: str | None, *, extracted_fact: str | None) -> bool:
    if not extracted_fact_matches_oracle(extracted_fact):
        return False
    combined = (answer or "").lower()
    return any(phrase.lower() in combined for phrase in EXPECTED_RELEASE_DATE_PHRASES)


def independent_web_oracle_passes(
    *,
    answer: str | None,
    actual_query: str | None,
    selected_source_ref: str | None,
    extracted_fact: str | None,
    candidate_refs: tuple[str, ...],
) -> bool:
    if actual_query is not None and not query_intent_matches(actual_query):
        return False
    if selected_source_ref is None:
        return False
    selected_url = url_from_artifact_ref(selected_source_ref)
    if not is_expected_python_3120_release_source(selected_url):
        return False
    if not official_source_present_in_candidates(candidate_refs):
        return False
    if not extracted_fact_matches_oracle(extracted_fact):
        return False
    return final_answer_matches_oracle(answer, extracted_fact=extracted_fact)


def _correlation(scope: PipelineEvidenceScope) -> DiagnosticExecutionCorrelation:
    return DiagnosticExecutionCorrelation(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
    )


def build_query_validation_evidence(
    scope: PipelineEvidenceScope,
    *,
    actual_query: str | None,
    idempotency_key: str,
) -> FunctionalValidationEvidence:
    passed = query_intent_matches(actual_query or "")
    validation_id = functional_validation_evidence_id(
        validator_id=Q3_QUERY_ORACLE_VALIDATOR_ID,
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        correlation=_correlation(scope),
        idempotency_key=f"{idempotency_key}:query",
    )
    return FunctionalValidationEvidence(
        validation_id=validation_id,
        validator=FunctionalValidatorRef(validator_id=Q3_QUERY_ORACLE_VALIDATOR_ID),
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        outcome=FunctionalValidationOutcome.PASSED if passed else FunctionalValidationOutcome.FAILED,
        correlation=_correlation(scope),
        expected_actual_relation=ExpectedActualRelation.CONTAINS,
        relation_summary=f"expected_intent=python_3.12 actual={(actual_query or '')[:200]}",
    )


def build_extraction_validation_evidence(
    scope: PipelineEvidenceScope,
    *,
    extracted_fact: str | None,
    idempotency_key: str,
) -> FunctionalValidationEvidence:
    passed = extracted_fact_matches_oracle(extracted_fact)
    validation_id = functional_validation_evidence_id(
        validator_id=Q3_EXTRACTION_ORACLE_VALIDATOR_ID,
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        correlation=_correlation(scope),
        idempotency_key=f"{idempotency_key}:extraction",
    )
    return FunctionalValidationEvidence(
        validation_id=validation_id,
        validator=FunctionalValidatorRef(validator_id=Q3_EXTRACTION_ORACLE_VALIDATOR_ID),
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        outcome=FunctionalValidationOutcome.PASSED if passed else FunctionalValidationOutcome.FAILED,
        correlation=_correlation(scope),
        expected_actual_relation=ExpectedActualRelation.CONTAINS,
        relation_summary=f"expected_release_date actual={(extracted_fact or '')[:200]}",
    )


def build_final_validation_evidence(
    scope: PipelineEvidenceScope,
    *,
    answer: str | None,
    extracted_fact: str | None,
    idempotency_key: str,
) -> FunctionalValidationEvidence:
    passed = final_answer_matches_oracle(answer, extracted_fact=extracted_fact)
    validation_id = functional_validation_evidence_id(
        validator_id=Q3_ORACLE_VALIDATOR_ID,
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        correlation=_correlation(scope),
        idempotency_key=f"{idempotency_key}:final",
    )
    return FunctionalValidationEvidence(
        validation_id=validation_id,
        validator=FunctionalValidatorRef(validator_id=Q3_ORACLE_VALIDATOR_ID),
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        outcome=FunctionalValidationOutcome.PASSED if passed else FunctionalValidationOutcome.FAILED,
        correlation=_correlation(scope),
        expected_actual_relation=ExpectedActualRelation.CONTAINS,
        relation_summary=f"expected_release_date actual={(answer or '')[:200]}",
    )


def bounded_provider_results(candidate_refs: tuple[str, ...]) -> list[dict[str, str]]:
    return [
        {"rank": str(index + 1), "artifact_ref": ref, "url": url_from_artifact_ref(ref)}
        for index, ref in enumerate(candidate_refs[:8])
    ]


__all__ = [
    "CANONICAL_EXPECTED_SOURCE_REF",
    "EXPECTED_RELEASE_DATE_PHRASES",
    "HEALTHY_TASK",
    "Q3_EXTRACTION_ORACLE_VALIDATOR_ID",
    "Q3_ORACLE_VALIDATOR_ID",
    "Q3_QUERY_ORACLE_VALIDATOR_ID",
    "bounded_provider_results",
    "build_extraction_validation_evidence",
    "build_final_validation_evidence",
    "build_query_validation_evidence",
    "extracted_fact_matches_oracle",
    "independent_web_oracle_passes",
    "official_source_present_in_candidates",
    "query_intent_matches",
    "resolve_expected_official_source_ref",
]
