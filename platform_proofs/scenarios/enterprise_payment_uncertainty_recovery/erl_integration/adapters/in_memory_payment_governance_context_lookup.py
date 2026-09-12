"""In-memory payment governance business context for unit tests and local wiring."""

from __future__ import annotations

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.failures import (
    PaymentGovernanceContextMissing,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_governance_context import (
    PaymentGovernanceBusinessContext,
    PaymentGovernanceBusinessContextLookupPort,
)


class InMemoryPaymentGovernanceBusinessContextLookup(
    PaymentGovernanceBusinessContextLookupPort,
):
    def __init__(self) -> None:
        self._records: dict[str, PaymentGovernanceBusinessContext] = {}

    def seed(self, record: PaymentGovernanceBusinessContext) -> None:
        self._records[record.correlation_id.strip()] = record

    def lookup_by_correlation_id(self, correlation_id: str) -> PaymentGovernanceBusinessContext:
        normalized = correlation_id.strip()
        if not normalized:
            raise PaymentGovernanceContextMissing("empty_correlation_id")
        record = self._records.get(normalized)
        if record is None:
            raise PaymentGovernanceContextMissing(
                f"no_governance_context_for:{normalized}",
            )
        return record
