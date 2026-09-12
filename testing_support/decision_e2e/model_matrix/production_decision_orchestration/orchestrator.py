# © Artur Czarnecki. All rights reserved.

"""Production decision flow coordination (DS-E2E-15J-L6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing.contracts import (
    GovernanceDisposition,
    GovernanceEvaluationRequest,
)
from testing_support.decision_e2e.model_matrix.production_decision_orchestration.contracts import (
    ORCHESTRATION_TASK_ID,
    ORCHESTRATION_VERSION,
    DecisionExecutionRequest,
    DecisionOrchestrationLifecycleMetadata,
    DecisionOrchestrationLifecycleStage,
    DecisionOrchestrationOutcome,
    DecisionOrchestrationRequest,
    DecisionOrchestrationResult,
)
from testing_support.decision_e2e.model_matrix.production_decision_orchestration.errors import (
    DecisionOrchestrationProviderMissingError,
)
from testing_support.decision_e2e.model_matrix.production_decision_orchestration.protocol import (
    DecisionSelectionProvider,
    ExecutionProvider,
    GovernanceDecisionProvider,
)


def _require_selection_provider(
    provider: DecisionSelectionProvider | None,
) -> DecisionSelectionProvider:
    if provider is None:
        raise DecisionOrchestrationProviderMissingError(
            "selection provider is required for decision orchestration"
        )
    return provider


def _require_governance_provider(
    provider: GovernanceDecisionProvider | None,
) -> GovernanceDecisionProvider:
    if provider is None:
        raise DecisionOrchestrationProviderMissingError(
            "governance provider is required for decision orchestration"
        )
    return provider


def _require_execution_provider(
    provider: ExecutionProvider | None,
) -> ExecutionProvider:
    if provider is None:
        raise DecisionOrchestrationProviderMissingError(
            "execution provider is required for decision orchestration"
        )
    return provider


@dataclass(frozen=True, slots=True)
class DecisionOrchestrator:
    """Coordinates selection, governance, and execution without domain policy logic."""

    selection_provider: DecisionSelectionProvider | None
    governance_provider: GovernanceDecisionProvider | None
    execution_provider: ExecutionProvider | None

    def orchestrate(
        self,
        request: DecisionOrchestrationRequest,
        *,
        orchestrated_at: datetime | None = None,
    ) -> DecisionOrchestrationResult:
        selection = _require_selection_provider(self.selection_provider)
        governance = _require_governance_provider(self.governance_provider)
        execution = _require_execution_provider(self.execution_provider)

        stamp = orchestrated_at or datetime.now(tz=UTC)
        scenario_id = request.governance_task_context.scenario_id

        selection_result = selection.recommend(
            request.selection_request,
            recommended_at=stamp,
        )
        stages: list[DecisionOrchestrationLifecycleStage] = [
            DecisionOrchestrationLifecycleStage.SELECTED,
        ]

        governance_request = GovernanceEvaluationRequest(
            model_recommendation=selection_result,
            task_context=request.governance_task_context,
            applicable_policies=request.applicable_policies,
            capability_evidence=request.capability_evidence,
        )
        governance_result = governance.evaluate(
            governance_request,
            evaluated_at=stamp,
        )

        if governance_result.disposition is GovernanceDisposition.BLOCK:
            stages.extend(
                (
                    DecisionOrchestrationLifecycleStage.BLOCKED,
                    DecisionOrchestrationLifecycleStage.STOPPED,
                )
            )
            return DecisionOrchestrationResult(
                outcome=DecisionOrchestrationOutcome.GOVERNANCE_BLOCKED,
                selection_result=selection_result,
                governance_result=governance_result,
                execution_result_reference=None,
                lifecycle_metadata=DecisionOrchestrationLifecycleMetadata(
                    orchestration_task_id=ORCHESTRATION_TASK_ID,
                    orchestration_version=ORCHESTRATION_VERSION,
                    orchestrated_at=stamp,
                    scenario_id=scenario_id,
                    lifecycle_stages=tuple(stages),
                    selection_provider_id=selection.provider_id,
                    governance_provider_id=governance.provider_id,
                    execution_provider_id=None,
                ),
            )

        if governance_result.disposition is GovernanceDisposition.REQUIRE_APPROVAL:
            stages.extend(
                (
                    DecisionOrchestrationLifecycleStage.REQUIRE_APPROVAL,
                    DecisionOrchestrationLifecycleStage.STOPPED,
                )
            )
            return DecisionOrchestrationResult(
                outcome=DecisionOrchestrationOutcome.APPROVAL_REQUIRED,
                selection_result=selection_result,
                governance_result=governance_result,
                execution_result_reference=None,
                lifecycle_metadata=DecisionOrchestrationLifecycleMetadata(
                    orchestration_task_id=ORCHESTRATION_TASK_ID,
                    orchestration_version=ORCHESTRATION_VERSION,
                    orchestrated_at=stamp,
                    scenario_id=scenario_id,
                    lifecycle_stages=tuple(stages),
                    selection_provider_id=selection.provider_id,
                    governance_provider_id=governance.provider_id,
                    execution_provider_id=None,
                ),
            )

        stages.append(DecisionOrchestrationLifecycleStage.ALLOWED)
        execution_request = DecisionExecutionRequest(
            selection_result=selection_result,
            governance_decision=governance_result,
        )
        execution_reference = execution.execute(
            execution_request,
            executed_at=stamp,
        )
        stages.append(DecisionOrchestrationLifecycleStage.EXECUTED)

        return DecisionOrchestrationResult(
            outcome=DecisionOrchestrationOutcome.SUCCESS,
            selection_result=selection_result,
            governance_result=governance_result,
            execution_result_reference=execution_reference,
            lifecycle_metadata=DecisionOrchestrationLifecycleMetadata(
                orchestration_task_id=ORCHESTRATION_TASK_ID,
                orchestration_version=ORCHESTRATION_VERSION,
                orchestrated_at=stamp,
                scenario_id=scenario_id,
                lifecycle_stages=tuple(stages),
                selection_provider_id=selection.provider_id,
                governance_provider_id=governance.provider_id,
                execution_provider_id=execution.provider_id,
            ),
        )


__all__ = ["DecisionOrchestrator"]
