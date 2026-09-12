"""Outbound integration contract — response visible to the commerce application."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.failures import (
    CommunicationFailureKind,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.lifecycle import (
    ExternalPaymentLifecycleState,
    IntegrationResponseState,
)


@dataclass(frozen=True, slots=True)
class PaymentProcessingResult:
    """Processing outcome on the integration channel (not SoR terminal truth)."""

    correlation_id: str
    external_reference: str
    integration_status: IntegrationResponseState
    external_lifecycle_state: ExternalPaymentLifecycleState
    request_timestamp: datetime
    processing_timestamp: datetime | None
    communication_failure_kind: CommunicationFailureKind | None
