# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Q2 tool-selection functional diagnostic specification (DIAG-FUNCTIONAL-Q2)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import EventId
from intergrax.contracts.runtime_execution_context import WORKSPACE_WRITE_FILE_TOOL_ID
from intergrax.tools.providers.workspace.service import WORKSPACE_SEARCH_TOOL_ID
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
    SelectionArtifactMatchRequirement,
    ValidationOutcomeRequirement,
    validate_functional_diagnostic_specification,
)
from intergrax.runtime.diagnostics.functional_evidence import PipelineOperationStatus
from intergrax.runtime.observability.functional_validation_evidence import FunctionalValidationOutcome

Q2_TOOL_SPECIFICATION_ID = FunctionalDiagnosticSpecificationId(
    "fdspec_000000000000000000000000a2b20001",
)
Q2_TOOL_SPECIFICATION_VERSION = 1
Q2_TOOL_QUERY_ID = "q2-tool-selection-1"
Q2_TOOL_INVOKE_OPERATION_ID = "tool.invoke"
Q2_EXPECTED_SEARCH_TOOL_ARTIFACT = f"tool:{WORKSPACE_SEARCH_TOOL_ID}"
Q2_EXPECTED_WRITE_TOOL_ARTIFACT = f"tool:{WORKSPACE_WRITE_FILE_TOOL_ID}"

CHECK_Q2_CANDIDATES = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000011")
CHECK_Q2_SELECTION = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000012")
CHECK_Q2_INVOCATION = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000013")
CHECK_Q2_VALIDATION = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000014")


def build_q2_tool_selection_functional_diagnostic_specification(
    *,
    validation_id: EventId | None = None,
    include_validation: bool = True,
    expected_selection_artifact_ref: str = Q2_EXPECTED_SEARCH_TOOL_ARTIFACT,
) -> FunctionalDiagnosticSpecification:
    checks: list[FunctionalDiagnosticCheck] = [
        FunctionalDiagnosticCheck(
            check_id=CHECK_Q2_CANDIDATES,
            requirement=FunctionalDiagnosticRequirement(
                kind=FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS,
                candidate_exists=CandidateExistsRequirement(query_id=Q2_TOOL_QUERY_ID),
            ),
            pass_claim="Tool candidates were recorded.",
            fail_claim="No tool candidates were recorded.",
            insufficient_claim="No candidate rank evidence.",
        ),
        FunctionalDiagnosticCheck(
            check_id=CHECK_Q2_SELECTION,
            requirement=FunctionalDiagnosticRequirement(
                kind=FunctionalDiagnosticRequirementKind.SELECTION_ARTIFACT_MATCH,
                selection_artifact_match=SelectionArtifactMatchRequirement(
                    query_id=Q2_TOOL_QUERY_ID,
                    expected_artifact_ref=expected_selection_artifact_ref,
                ),
            ),
            pass_claim="Expected catalog tool was selected.",
            fail_claim="Wrong catalog tool was selected.",
            insufficient_claim="No tool selection evidence.",
        ),
        FunctionalDiagnosticCheck(
            check_id=CHECK_Q2_INVOCATION,
            requirement=FunctionalDiagnosticRequirement(
                kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                operation_outcome_status=OperationOutcomeStatusRequirement(
                    operation_id=Q2_TOOL_INVOKE_OPERATION_ID,
                    expected_status=PipelineOperationStatus.SUCCEEDED,
                ),
            ),
            pass_claim="Selected tool invocation succeeded.",
            fail_claim="Selected tool invocation failed.",
            insufficient_claim="No invocation outcome evidence.",
        ),
    ]
    if include_validation:
        if validation_id is None:
            raise ValueError("validation_id is required when include_validation is True")
        checks.append(
            FunctionalDiagnosticCheck(
                check_id=CHECK_Q2_VALIDATION,
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
            specification_id=Q2_TOOL_SPECIFICATION_ID,
            version=Q2_TOOL_SPECIFICATION_VERSION,
            checks=tuple(checks),
        ),
    )


__all__ = [
    "CHECK_Q2_CANDIDATES",
    "CHECK_Q2_INVOCATION",
    "CHECK_Q2_SELECTION",
    "CHECK_Q2_VALIDATION",
    "Q2_EXPECTED_SEARCH_TOOL_ARTIFACT",
    "Q2_EXPECTED_WRITE_TOOL_ARTIFACT",
    "Q2_TOOL_INVOKE_OPERATION_ID",
    "Q2_TOOL_QUERY_ID",
    "Q2_TOOL_SPECIFICATION_ID",
    "Q2_TOOL_SPECIFICATION_VERSION",
    "build_q2_tool_selection_functional_diagnostic_specification",
]
