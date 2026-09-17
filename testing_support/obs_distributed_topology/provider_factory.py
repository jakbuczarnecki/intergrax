# © Artur Czarnecki. All rights reserved.

"""Provider-neutral evidence persistence factory seam (OBS-DG005)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.runtime.events.evidence_persistence_adapter import (
    as_evidence_persistence_port,
)
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore

from testing_support.obs_distributed_topology.models import SqliteEvidenceProviderConfig

EvidenceProviderFactory = Callable[[SqliteEvidenceProviderConfig], EvidencePersistencePort]


def sqlite_file_evidence_provider_factory(
    config: SqliteEvidenceProviderConfig,
) -> EvidencePersistencePort:
    store = SQLiteRuntimeEventStore(db_path=Path(config.db_path))
    return as_evidence_persistence_port(store)


DEFAULT_DG005_EVIDENCE_PROVIDER_FACTORY: EvidenceProviderFactory = (
    sqlite_file_evidence_provider_factory
)
