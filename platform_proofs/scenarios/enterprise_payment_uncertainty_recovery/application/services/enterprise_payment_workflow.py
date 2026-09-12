"""Enterprise payment workflow — order exists, then payment capture is requested."""

from __future__ import annotations

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.scenario_context import (
    ScenarioExecutionContext,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.domain.workflow import (
    BusinessWorkflowPhase,
    PaymentWorkflowOutcome,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.observability import (
    BusinessActionKind,
    ScenarioApplicationObservability,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.ports.order_access import (
    OrderAccessPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.ports.payment_workflow import (
    PaymentWorkflowPort,
)


class EnterprisePaymentWorkflowService:
    """Coordinates domain actions for the lab purchase-order payment path."""

    def __init__(
        self,
        *,
        order_access: OrderAccessPort,
        payment_workflow: PaymentWorkflowPort,
        observability: ScenarioApplicationObservability,
    ) -> None:
        self._order_access = order_access
        self._payment_workflow = payment_workflow
        self._observability = observability

    def execute(self, context: ScenarioExecutionContext) -> PaymentWorkflowOutcome:
        order = self._order_access.load_order_for_execution(context)
        self._observability.business_action_executed(
            context,
            kind=BusinessActionKind.ORDER_LOADED,
            detail={"order_number": order.order_number, "business_status": order.business_status},
        )
        payment_request = self._payment_workflow.request_capture(context, order)
        self._observability.business_action_executed(
            context,
            kind=BusinessActionKind.PAYMENT_CAPTURE_REQUESTED,
            detail={
                "intent_reference": payment_request.intent_reference,
                "order_number": order.order_number,
            },
        )
        return PaymentWorkflowOutcome(
            phase=BusinessWorkflowPhase.PAYMENT_REQUESTED,
            order=order,
            payment_request=payment_request,
        )
