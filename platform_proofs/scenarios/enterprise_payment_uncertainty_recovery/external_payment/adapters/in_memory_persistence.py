"""In-process SoR store for boundary tests and lab wiring without PostgreSQL."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.contracts.persistence import (
    ExternalRealityPersistenceBundle,
    ExternalRealityPersistencePort,
)


@dataclass(frozen=True, slots=True)
class StoredExternalReality:
    bundle: ExternalRealityPersistenceBundle


class InMemoryExternalRealityStore(ExternalRealityPersistencePort):
    """Retains persisted external effects and reality for assertions."""

    def __init__(self) -> None:
        self.records: list[StoredExternalReality] = []

    def persist_external_reality(self, bundle: ExternalRealityPersistenceBundle) -> None:
        self.records.append(StoredExternalReality(bundle=bundle))

    def latest(self) -> StoredExternalReality | None:
        if not self.records:
            return None
        return self.records[-1]
