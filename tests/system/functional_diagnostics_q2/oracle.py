# © Artur Czarnecki. All rights reserved.

"""Independent deterministic oracle for Q2 tool-selection qualification."""

from __future__ import annotations

from intergrax.tools.providers.workspace.service import WORKSPACE_SEARCH_TOOL_ID
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

Q2_ORACLE_VALIDATOR_ID = "q2.tool.functional_oracle.v1"
EXPECTED_INCIDENT_DATE = "2026-08-17"
EXPECTED_SEARCH_TOOL_ARTIFACT = f"tool:{WORKSPACE_SEARCH_TOOL_ID}"


def independent_tool_oracle_passes(
    *,
    answer: str | None,
    selected_tool_artifact: str | None,
) -> bool:
    if selected_tool_artifact != EXPECTED_SEARCH_TOOL_ARTIFACT:
        return False
    combined = answer or ""
    return EXPECTED_INCIDENT_DATE in combined


def build_independent_validation_evidence(
    scope: PipelineEvidenceScope,
    *,
    answer: str | None,
    selected_tool_artifact: str | None,
    idempotency_key: str,
) -> FunctionalValidationEvidence:
    passed = independent_tool_oracle_passes(
        answer=answer,
        selected_tool_artifact=selected_tool_artifact,
    )
    correlation = DiagnosticExecutionCorrelation(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
    )
    validation_id = functional_validation_evidence_id(
        validator_id=Q2_ORACLE_VALIDATOR_ID,
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        correlation=correlation,
        idempotency_key=idempotency_key,
    )
    return FunctionalValidationEvidence(
        validation_id=validation_id,
        validator=FunctionalValidatorRef(validator_id=Q2_ORACLE_VALIDATOR_ID),
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        outcome=(
            FunctionalValidationOutcome.PASSED if passed else FunctionalValidationOutcome.FAILED
        ),
        correlation=correlation,
        expected_actual_relation=ExpectedActualRelation.CONTAINS,
    )


__all__ = [
    "EXPECTED_INCIDENT_DATE",
    "EXPECTED_SEARCH_TOOL_ARTIFACT",
    "Q2_ORACLE_VALIDATOR_ID",
    "build_independent_validation_evidence",
    "independent_tool_oracle_passes",
]
