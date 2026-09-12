"""Scenario application composition root — startup boundary without business rules."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.dependencies import (
    ApplicationDependencies,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.scenario_context import (
    ScenarioExecutionContext,
    validate_scenario_execution_context,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.domain.workflow import (
    PaymentWorkflowOutcome,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.services.enterprise_payment_workflow import (
    EnterprisePaymentWorkflowService,
)


@dataclass(frozen=True, slots=True)
class ScenarioApplicationRunResult:
    """Outcome of one application execution (business skeleton stage)."""

    context: ScenarioExecutionContext
    workflow_outcome: PaymentWorkflowOutcome


class ScenarioApplicationCompositionRoot:
    """Initialize ports and execute the enterprise payment workflow for one run."""

    def __init__(self, dependencies: ApplicationDependencies) -> None:
        self._dependencies = dependencies
        self._workflow = EnterprisePaymentWorkflowService(
            order_access=dependencies.order_access,
            payment_workflow=dependencies.payment_workflow,
            observability=dependencies.observability,
        )

    def run(self, context: ScenarioExecutionContext) -> ScenarioApplicationRunResult:
        validate_scenario_execution_context(context)
        self._dependencies.observability.scenario_started(context)
        workflow_outcome = self._workflow.execute(context)
        self._dependencies.observability.scenario_state_prepared(context, workflow_outcome)
        return ScenarioApplicationRunResult(context=context, workflow_outcome=workflow_outcome)
