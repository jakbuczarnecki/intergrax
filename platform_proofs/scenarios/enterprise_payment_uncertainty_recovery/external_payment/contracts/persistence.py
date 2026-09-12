"""Persistence port for external system-of-record artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.lifecycle import (
    ExternalPaymentLifecycleState,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.sor_truth import (
    SorTruthFields,
)


@dataclass(frozen=True, slots=True)
class ExternalRealityPersistenceBundle:
    external_payment_effect_id: UUID
    external_reality_id: UUID
    payment_intent_id: UUID
    external_effect_reference: str
    correlation_id: str
    sor_transaction_ref: str
    requested_state: str
    observed_integration_state: str
    lifecycle_state: ExternalPaymentLifecycleState
    sor_truth: SorTruthFields
    requested_at: datetime
    processed_at: datetime


class ExternalRealityPersistencePort(Protocol):
    def persist_external_reality(self, bundle: ExternalRealityPersistenceBundle) -> None:
        """Record external payment effect and authoritative reality (SoR ownership)."""
        ...
