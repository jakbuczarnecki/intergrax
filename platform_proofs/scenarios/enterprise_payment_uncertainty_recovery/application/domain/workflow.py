"""Business workflow phases — order exists through payment requested."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.domain.entities import (
    OrderSnapshot,
    PaymentCaptureRequest,
)


class BusinessWorkflowPhase(StrEnum):
    ORDER_LOADED = "order_loaded"
    PAYMENT_REQUESTED = "payment_requested"


@dataclass(frozen=True, slots=True)
class PaymentWorkflowOutcome:
    """Terminal state for the current application skeleton (payment requested, not resolved)."""

    phase: BusinessWorkflowPhase
    order: OrderSnapshot
    payment_request: PaymentCaptureRequest
