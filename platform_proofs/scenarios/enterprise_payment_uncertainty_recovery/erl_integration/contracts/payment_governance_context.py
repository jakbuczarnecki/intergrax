"""Scenario-owned payment governance inputs — not part of platform contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol


@dataclass(frozen=True, slots=True)
class PaymentEnterpriseGovernancePolicy:
    """Injectable enterprise thresholds — no global registry."""

    human_approval_threshold_amount: Decimal
    currency: str = "EUR"


@dataclass(frozen=True, slots=True)
class PaymentGovernanceBusinessContext:
    """Business attributes used to evaluate payment continuation governance."""

    correlation_id: str
    payment_amount: Decimal | None
    currency: str
    customer_risk_tier: str | None = None


class PaymentGovernanceBusinessContextLookupPort(Protocol):
    """Replaceable order / payment business context (PostgreSQL lab, in-memory tests)."""

    def lookup_by_correlation_id(
        self,
        correlation_id: str,
    ) -> PaymentGovernanceBusinessContext:
        """Return governance business context for the correlation id."""
        ...
