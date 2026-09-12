"""Bridge external payment persistence to ERL reconciliation lookup ports."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.adapters.in_memory_external_reality_lookup import (
    InMemoryExternalRealityLookup,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.adapters.in_memory_payment_evidence_lookup import (
    InMemoryPaymentReconciliationEvidenceLookup,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.external_reality_lookup import (
    ExternalRealitySnapshot,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.failures import (
    ExternalRealityRecordMissing,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.mapping.payment_evidence_fields import (
    resolve_payment_reconciliation_evidence,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.adapters.in_memory_persistence import (
    InMemoryExternalRealityStore,
)


def load_variant_document(dataset_package_root: Path, variant_id: str) -> dict:
    path = dataset_package_root / "variants" / variant_id / "scenario_variant.json"
    return json.loads(path.read_text(encoding="utf-8"))


class PersistedExternalRealityLookup(InMemoryExternalRealityLookup):
    """Seeds reconciliation lookup from capture persistence — facts, not variant labels."""

    def __init__(self, store: InMemoryExternalRealityStore) -> None:
        super().__init__()
        self._store = store

    def refresh_from_persistence(self, correlation_id: str) -> ExternalRealitySnapshot:
        normalized = correlation_id.strip()
        for stored in reversed(self._store.records):
            bundle = stored.bundle
            if bundle.correlation_id.strip() != normalized:
                continue
            snapshot = ExternalRealitySnapshot(
                correlation_id=bundle.correlation_id,
                external_effect_reference=bundle.external_effect_reference,
                terminal_outcome=bundle.sor_truth.terminal_outcome,
                funds_captured=bundle.sor_truth.funds_captured,
                truth_availability_state=bundle.sor_truth.truth_availability_state,
                sor_transaction_ref=bundle.sor_transaction_ref,
            )
            self.seed(snapshot)
            return snapshot
        raise ExternalRealityRecordMissing(f"no_persisted_external_reality_for:{normalized}")


def build_payment_evidence_lookup(
    *,
    dataset_package_root: Path,
    variant_id: str,
    snapshot: ExternalRealitySnapshot,
    observed_at: datetime | None = None,
) -> InMemoryPaymentReconciliationEvidenceLookup:
    variant_document = load_variant_document(dataset_package_root, variant_id)
    payment = resolve_payment_reconciliation_evidence(
        variant_document,
        correlation_id=snapshot.correlation_id,
        external_effect_reference=snapshot.external_effect_reference,
        sor_transaction_ref=snapshot.sor_transaction_ref,
        funds_captured=snapshot.funds_captured,
        truth_availability_state=snapshot.truth_availability_state,
        observed_at=observed_at or datetime.now(tz=UTC),
    )
    lookup = InMemoryPaymentReconciliationEvidenceLookup()
    lookup.seed(payment)
    return lookup
