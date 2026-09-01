# © Artur Czarnecki. All rights reserved.

"""Independent deterministic oracle for C1 RAG qualification (DIAG-FUNCTIONAL-Q1)."""

from __future__ import annotations

from intergrax.runtime.diagnostics.functional_validation import (
    DiagnosticExecutionCorrelation,
    ExpectedActualRelation,
    FunctionalValidationEvidence,
    FunctionalValidationKind,
    FunctionalValidationOutcome,
    FunctionalValidatorRef,
    functional_validation_evidence_id,
)
from intergrax.runtime.diagnostics.functional_evidence import PipelineEvidenceScope

C1_ORACLE_VALIDATOR_ID = "c1.rag.date_oracle.v1"
EXPECTED_INCIDENT_DATE = "2026-08-17"
SEARCH_QUESTION = "When did Incident Orion occur?"
_FORCED_WRONG_SELECTION_REFS = frozenset({"chunk:operations-decoy"})


def search_request_message() -> str:
    return SEARCH_QUESTION


def evidence_texts_from_lkw_response(
    *,
    answer: str | None,
    lkw_evidence: dict[str, object] | None,
) -> list[str]:
    texts: list[str] = []
    if answer:
        texts.append(answer)
    if not isinstance(lkw_evidence, dict):
        return texts
    diagnostics = lkw_evidence.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return texts
    search_diag = diagnostics.get("lkw.search_summary.v1")
    if isinstance(search_diag, dict):
        source_refs = search_diag.get("source_refs")
        if isinstance(source_refs, list):
            texts.extend(str(item) for item in source_refs)
    return texts


def independent_date_oracle_passes(
    *,
    answer: str | None,
    evidence_texts: list[str],
) -> bool:
    combined = "\n".join([answer or "", *evidence_texts])
    if EXPECTED_INCIDENT_DATE in combined:
        return True
    lowered = combined.lower()
    return "incident-report" in lowered and "orion" in lowered


def resolve_qualification_functional_pass(
    *,
    metadata: dict[str, object],
    answer: str | None,
    evidence_texts: list[str],
) -> bool:
    """Independent oracle using qualification fixture config, not DIAG output."""
    force = metadata.get("qualification_force_selection_artifact_ref")
    if isinstance(force, str) and force.strip() in _FORCED_WRONG_SELECTION_REFS:
        return False
    draft_override = metadata.get("qualification_draft_override")
    if isinstance(draft_override, str) and draft_override.strip():
        return independent_date_oracle_passes(
            answer=draft_override,
            evidence_texts=[],
        )
    return independent_date_oracle_passes(answer=answer, evidence_texts=evidence_texts)


def build_independent_validation_evidence(
    scope: PipelineEvidenceScope,
    *,
    answer: str | None,
    evidence_texts: list[str],
    idempotency_key: str,
    metadata: dict[str, object] | None = None,
) -> FunctionalValidationEvidence:
    passed = (
        resolve_qualification_functional_pass(
            metadata=metadata or {},
            answer=answer,
            evidence_texts=evidence_texts,
        )
        if metadata is not None
        else independent_date_oracle_passes(answer=answer, evidence_texts=evidence_texts)
    )
    correlation = DiagnosticExecutionCorrelation(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
    )
    validation_id = functional_validation_evidence_id(
        validator_id=C1_ORACLE_VALIDATOR_ID,
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        correlation=correlation,
        idempotency_key=idempotency_key,
    )
    return FunctionalValidationEvidence(
        validation_id=validation_id,
        validator=FunctionalValidatorRef(validator_id=C1_ORACLE_VALIDATOR_ID),
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        outcome=(
            FunctionalValidationOutcome.PASSED
            if passed
            else FunctionalValidationOutcome.FAILED
        ),
        correlation=correlation,
        expected_actual_relation=ExpectedActualRelation.CONTAINS,
        relation_summary=(
            f"expected={EXPECTED_INCIDENT_DATE} "
            f"actual={(answer or '')[:200]}"
        ),
    )


__all__ = [
    "C1_ORACLE_VALIDATOR_ID",
    "EXPECTED_INCIDENT_DATE",
    "SEARCH_QUESTION",
    "build_independent_validation_evidence",
    "evidence_texts_from_lkw_response",
    "independent_date_oracle_passes",
    "resolve_qualification_functional_pass",
    "search_request_message",
]
