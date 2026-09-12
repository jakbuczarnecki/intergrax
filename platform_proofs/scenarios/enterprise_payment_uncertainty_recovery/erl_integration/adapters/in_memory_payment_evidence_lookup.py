"""In-memory payment reconciliation evidence for unit tests and local wiring."""

from __future__ import annotations

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.failures import (
    PaymentReconciliationEvidenceMissing,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_reconciliation_evidence import (
    PaymentReconciliationEvidence,
    PaymentReconciliationEvidenceLookupPort,
)


class InMemoryPaymentReconciliationEvidenceLookup(PaymentReconciliationEvidenceLookupPort):
    def __init__(self) -> None:
        self._records: dict[str, PaymentReconciliationEvidence] = {}

    def seed(self, record: PaymentReconciliationEvidence) -> None:
        self._records[record.correlation_id.strip()] = record

    def lookup_by_correlation_id(self, correlation_id: str) -> PaymentReconciliationEvidence:
        normalized = correlation_id.strip()
        if not normalized:
            raise PaymentReconciliationEvidenceMissing("empty_correlation_id")
        record = self._records.get(normalized)
        if record is None:
            raise PaymentReconciliationEvidenceMissing(
                f"no_payment_evidence_for:{normalized}",
            )
        return record
