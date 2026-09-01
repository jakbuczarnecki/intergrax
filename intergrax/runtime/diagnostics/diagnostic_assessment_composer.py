# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operator diagnostic assessment composition (DIAG-FUNCTIONAL-4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessment,
    DiagnosticAssessmentIntegrityError,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import (
    FunctionalDiagnosticAnalysis,
)
from intergrax.runtime.diagnostics.functional_operator_projection import (
    FunctionalDiagnosticOperatorProjection,
    FunctionalOperatorProjector,
)


@dataclass(frozen=True, slots=True)
class OperatorDiagnosticAssessment:
    """
    One operator view combining lifecycle and optional functional diagnostics.

    Lifecycle findings remain lifecycle-specific. Functional projection is a
    separate layer derived from ready ``FunctionalDiagnosticAnalysis``.
    Execution terminal state is owned by Execution System contracts and is
    not duplicated here.
    """

    lifecycle_assessment: DiagnosticAssessment
    functional_projection: FunctionalDiagnosticOperatorProjection | None

    @property
    def tenant_id(self) -> str:
        return self.lifecycle_assessment.tenant_id

    @property
    def task_id(self) -> TaskId:
        return self.lifecycle_assessment.task_id

    @property
    def run_id(self) -> RunId:
        return self.lifecycle_assessment.run_id

    @property
    def has_lifecycle_findings(self) -> bool:
        return self.lifecycle_assessment.has_findings

    @property
    def has_functional_projection(self) -> bool:
        return self.functional_projection is not None


class DiagnosticAssessmentComposer:
    """
    Deterministic composition over ready lifecycle and functional analyses.

    Does not query persistence, perform lifecycle/functional analysis, or
    infer root cause.
    """

    def __init__(self, *, functional_projector: FunctionalOperatorProjector | None = None) -> None:
        self._functional_projector = functional_projector or FunctionalOperatorProjector()

    def compose(
        self,
        lifecycle_assessment: DiagnosticAssessment,
        functional_analysis: FunctionalDiagnosticAnalysis | None = None,
    ) -> OperatorDiagnosticAssessment:
        if type(lifecycle_assessment) is not DiagnosticAssessment:
            raise TypeError("lifecycle_assessment must be DiagnosticAssessment")
        if functional_analysis is not None and type(functional_analysis) is not FunctionalDiagnosticAnalysis:
            raise TypeError("functional_analysis must be FunctionalDiagnosticAnalysis or None")

        functional_projection: FunctionalDiagnosticOperatorProjection | None = None
        if functional_analysis is not None:
            _validate_composition_scope(lifecycle_assessment, functional_analysis)
            functional_projection = self._functional_projector.project(functional_analysis)

        return OperatorDiagnosticAssessment(
            lifecycle_assessment=lifecycle_assessment,
            functional_projection=functional_projection,
        )


def _validate_composition_scope(
    lifecycle_assessment: DiagnosticAssessment,
    functional_analysis: FunctionalDiagnosticAnalysis,
) -> None:
    if lifecycle_assessment.tenant_id != functional_analysis.tenant_id:
        raise DiagnosticAssessmentIntegrityError(
            "lifecycle assessment tenant_id does not match functional analysis scope",
        )
    if lifecycle_assessment.task_id != functional_analysis.task_id:
        raise DiagnosticAssessmentIntegrityError(
            "lifecycle assessment task_id does not match functional analysis scope",
        )
    if lifecycle_assessment.run_id != functional_analysis.run_id:
        raise DiagnosticAssessmentIntegrityError(
            "lifecycle assessment run_id does not match functional analysis scope",
        )


__all__ = [
    "DiagnosticAssessmentComposer",
    "OperatorDiagnosticAssessment",
]
