# © Artur Czarnecki. All rights reserved.

"""Minimal functional diagnostic profile for MP-4R7 protected operation evidence."""

from __future__ import annotations

from intergrax.contracts.functional_evidence import PipelineOperationStatus
from intergrax.runtime.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId,
    FunctionalDiagnosticSpecificationId,
)
from intergrax.runtime.diagnostics.functional_diagnostic_specification import (
    FunctionalDiagnosticCheck,
    FunctionalDiagnosticRequirement,
    FunctionalDiagnosticRequirementKind,
    FunctionalDiagnosticSpecification,
    OperationOutcomeStatusRequirement,
    validate_functional_diagnostic_specification,
)

_MP4R7_SPEC_ID = FunctionalDiagnosticSpecificationId(
    "fdspec_00000000000000000000000470000100",
)
_MP4R7_CHECK_ID = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000470000100")


def mp4r7_protected_operation_diagnostic_specification() -> FunctionalDiagnosticSpecification:
    spec = FunctionalDiagnosticSpecification(
        specification_id=_MP4R7_SPEC_ID,
        version=1,
        checks=(
            FunctionalDiagnosticCheck(
                check_id=_MP4R7_CHECK_ID,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="mp4r7.enterprise.protected_side_effect",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Protected enterprise side effect recorded operation_outcome SUCCEEDED.",
                fail_claim="Protected enterprise side effect missing or not SUCCEEDED.",
                insufficient_claim="No operation_outcome evidence available for protected side effect.",
            ),
        ),
    )
    return validate_functional_diagnostic_specification(spec)


__all__ = ["mp4r7_protected_operation_diagnostic_specification"]
