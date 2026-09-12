"""Scenario-owned payment reconciliation evidence — not part of platform contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class PaymentReconciliationEvidence:
    """
    Enterprise payment attributes used to judge reconciliation quality.

    Populated from SoR, PSP confirmation, and settlement reads — never returned
    as business verdicts to the platform; only drives generic ERL outcomes.
    """

    correlation_id: str
    external_effect_reference: str
    psp_confirmation_id: str | None
    sor_transaction_ref: str | None
    reconciliation_availability: str
    discoverable_outcome: str | None
    funds_captured: bool
    settlement_status: str
    settlement_batch_ref: str | None
    source_reliability_tier: str
    evidence_observed_at: datetime | None


class PaymentReconciliationEvidenceLookupPort(Protocol):
    """Replaceable payment evidence source (PostgreSQL lab, in-memory tests)."""

    def lookup_by_correlation_id(
        self,
        correlation_id: str,
    ) -> PaymentReconciliationEvidence:
        """Return payment reconciliation attributes for the correlation id."""
        ...
