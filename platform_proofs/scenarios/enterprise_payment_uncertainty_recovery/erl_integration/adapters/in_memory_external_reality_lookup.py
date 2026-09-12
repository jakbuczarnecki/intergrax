"""In-memory external reality lookup for unit tests and local wiring."""

from __future__ import annotations

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.external_reality_lookup import (
    ExternalRealityLookupPort,
    ExternalRealitySnapshot,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.failures import (
    ExternalRealityRecordMissing,
    ExternalRealitySourceUnavailable,
)


class InMemoryExternalRealityLookup(ExternalRealityLookupPort):
    def __init__(self) -> None:
        self._snapshots: dict[str, ExternalRealitySnapshot] = {}
        self._available = True

    def seed(self, snapshot: ExternalRealitySnapshot) -> None:
        self._snapshots[snapshot.correlation_id.strip()] = snapshot

    def set_source_available(self, available: bool) -> None:
        self._available = available

    def lookup_by_correlation_id(self, correlation_id: str) -> ExternalRealitySnapshot:
        if not self._available:
            raise ExternalRealitySourceUnavailable("in_memory_source_marked_unavailable")
        normalized = correlation_id.strip()
        if not normalized:
            raise ExternalRealityRecordMissing("empty_correlation_id")
        snapshot = self._snapshots.get(normalized)
        if snapshot is None:
            raise ExternalRealityRecordMissing(f"no_external_reality_for:{normalized}")
        return snapshot
