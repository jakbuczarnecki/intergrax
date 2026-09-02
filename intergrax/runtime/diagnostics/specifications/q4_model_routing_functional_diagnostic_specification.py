# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Q4 model-routing functional diagnostic specification (DIAG-FUNCTIONAL-Q4)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import EventId
from intergrax.runtime.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId,
    FunctionalDiagnosticSpecificationId,
)
from intergrax.runtime.diagnostics.functional_diagnostic_specification import (
    CandidateExistsRequirement,
    FunctionalDiagnosticCheck,
    FunctionalDiagnosticRequirement,
    FunctionalDiagnosticRequirementKind,
    FunctionalDiagnosticSpecification,
    OperationOutcomeStatusRequirement,
    OutputRelationExistsRequirement,
    SelectionArtifactMatchRequirement,
    ValidationOutcomeRequirement,
    validate_functional_diagnostic_specification,
)
from intergrax.runtime.diagnostics.functional_evidence import PipelineOperationStatus
from intergrax.runtime.observability.functional_validation_evidence import FunctionalValidationOutcome

Q4_MODEL_SPECIFICATION_ID = FunctionalDiagnosticSpecificationId(
    "fdspec_000000000000000000000000a4b40001",
)
Q4_MODEL_SPECIFICATION_VERSION = 1
Q4_MODEL_QUERY_ID = "q4-model-routing-1"
Q4_MODEL_GENERATE_OPERATION_ID = "model.generate"

CHECK_Q4_CANDIDATES = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000031")
CHECK_Q4_SELECTION = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000032")
CHECK_Q4_INVOCATION = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000033")
CHECK_Q4_OUTPUT_RELATION = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000034")
CHECK_Q4_VALIDATION = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000035")


def build_q4_model_routing_functional_diagnostic_specification(
    *,
    validation_id: EventId | None = None,
    include_validation: bool = True,
    expected_selection_artifact_ref: str,
) -> FunctionalDiagnosticSpecification:
    checks: list[FunctionalDiagnosticCheck] = [
        FunctionalDiagnosticCheck(
            check_id=CHECK_Q4_CANDIDATES,
            requirement=FunctionalDiagnosticRequirement(
                kind=FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS,
                candidate_exists=CandidateExistsRequirement(query_id=Q4_MODEL_QUERY_ID),
            ),
            pass_claim="Routing candidate profiles were recorded.",
            fail_claim="No routing candidate profiles were recorded.",
            insufficient_claim="No candidate rank evidence.",
        ),
        FunctionalDiagnosticCheck(
            check_id=CHECK_Q4_SELECTION,
            requirement=FunctionalDiagnosticRequirement(
                kind=FunctionalDiagnosticRequirementKind.SELECTION_ARTIFACT_MATCH,
                selection_artifact_match=SelectionArtifactMatchRequirement(
                    query_id=Q4_MODEL_QUERY_ID,
                    expected_artifact_ref=expected_selection_artifact_ref,
                ),
            ),
            pass_claim="Expected LLM profile was selected.",
            fail_claim="Wrong LLM profile was selected.",
            insufficient_claim="No model selection evidence.",
        ),
        FunctionalDiagnosticCheck(
            check_id=CHECK_Q4_INVOCATION,
            requirement=FunctionalDiagnosticRequirement(
                kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                operation_outcome_status=OperationOutcomeStatusRequirement(
                    operation_id=Q4_MODEL_GENERATE_OPERATION_ID,
                    expected_status=PipelineOperationStatus.SUCCEEDED,
                ),
            ),
            pass_claim="Selected model invocation succeeded.",
            fail_claim="Selected model invocation failed.",
            insufficient_claim="No model invocation outcome evidence.",
        ),
        FunctionalDiagnosticCheck(
            check_id=CHECK_Q4_OUTPUT_RELATION,
            requirement=FunctionalDiagnosticRequirement(
                kind=FunctionalDiagnosticRequirementKind.OUTPUT_RELATION_EXISTS,
                output_relation_exists=OutputRelationExistsRequirement(
                    operation_id=Q4_MODEL_GENERATE_OPERATION_ID,
                ),
            ),
            pass_claim="Model output relation was recorded.",
            fail_claim="Model output relation missing.",
            insufficient_claim="No model output relation evidence.",
        ),
    ]
    if include_validation:
        if validation_id is None:
            raise ValueError("validation_id is required when include_validation is True")
        checks.append(
            FunctionalDiagnosticCheck(
                check_id=CHECK_Q4_VALIDATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.VALIDATION_OUTCOME,
                    validation_outcome=ValidationOutcomeRequirement(
                        validation_id=validation_id,
                        expected_outcome=FunctionalValidationOutcome.PASSED,
                    ),
                ),
                pass_claim="External oracle validation passed.",
                fail_claim="External oracle validation failed.",
                insufficient_claim="No linked validation evidence.",
            ),
        )
    return validate_functional_diagnostic_specification(
        FunctionalDiagnosticSpecification(
            specification_id=Q4_MODEL_SPECIFICATION_ID,
            version=Q4_MODEL_SPECIFICATION_VERSION,
            checks=tuple(checks),
        ),
    )


__all__ = [
    "CHECK_Q4_CANDIDATES",
    "CHECK_Q4_INVOCATION",
    "CHECK_Q4_OUTPUT_RELATION",
    "CHECK_Q4_SELECTION",
    "CHECK_Q4_VALIDATION",
    "Q4_MODEL_GENERATE_OPERATION_ID",
    "Q4_MODEL_QUERY_ID",
    "Q4_MODEL_SPECIFICATION_ID",
    "Q4_MODEL_SPECIFICATION_VERSION",
    "build_q4_model_routing_functional_diagnostic_specification",
]
