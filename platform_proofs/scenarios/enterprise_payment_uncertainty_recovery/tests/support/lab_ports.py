"""Test doubles implementing application ports — not part of the production composition path."""

from __future__ import annotations

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.adapters.lab_reference_order_access import (
    LabReferenceOrderAccess,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.scenario_context import (
    ScenarioExecutionContext,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.domain.entities import (
    OrderSnapshot,
    PaymentCaptureRequest,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.application.references import (
    LabBusinessReferences,
)


class LabReferencePaymentWorkflow:
    """Issues a realistic payment capture request without external I/O."""

    def __init__(self, references: LabBusinessReferences | None = None) -> None:
        self._references = references or LabBusinessReferences()

    def request_capture(
        self,
        context: ScenarioExecutionContext,
        order: OrderSnapshot,
    ) -> PaymentCaptureRequest:
        return PaymentCaptureRequest(
            intent_reference=self._references.payment_intent_reference,
            idempotency_key=self._references.payment_idempotency_key,
            related_order_number=order.order_number,
            amount=order.amount,
            currency=order.currency,
        )
