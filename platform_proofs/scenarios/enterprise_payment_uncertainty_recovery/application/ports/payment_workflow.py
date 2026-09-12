"""Payment workflow port — coordinates capture request without reliability decisions."""

from __future__ import annotations

from typing import Protocol

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.scenario_context import (
    ScenarioExecutionContext,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.domain.entities import (
    OrderSnapshot,
    PaymentCaptureRequest,
)


class PaymentWorkflowPort(Protocol):
    def request_capture(
        self,
        context: ScenarioExecutionContext,
        order: OrderSnapshot,
    ) -> PaymentCaptureRequest:
        """Initiate payment capture for the order (integration boundary — no ERL logic)."""
        ...
