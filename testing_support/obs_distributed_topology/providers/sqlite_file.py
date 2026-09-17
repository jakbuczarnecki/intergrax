# © Artur Czarnecki. All rights reserved.

"""SQLite file-backed qualification evidence provider (OBS-DG005 default)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from intergrax.contracts.execution_evidence.persistence_port import (
    EvidencePersistencePort,
)
from intergrax.knowledge.contracts.validation import JsonValue
from intergrax.runtime.events.evidence_persistence_adapter import (
    as_evidence_persistence_port,
)
from intergrax.runtime.events.stores.sqlite_runtime_event_store import (
    SQLiteRuntimeEventStore,
)

from testing_support.obs_distributed_topology.provider_contract import (
    EvidenceProviderDescriptor,
)

SQLITE_FILE_PROVIDER_ID = "sqlite-file"
_DB_PATH_CONFIG_KEY = "db_path"


def sqlite_file_descriptor(*, db_path: Path | str) -> EvidenceProviderDescriptor:
    return EvidenceProviderDescriptor(
        provider_id=SQLITE_FILE_PROVIDER_ID,
        config={_DB_PATH_CONFIG_KEY: str(db_path)},
    )


def _require_db_path(config: Mapping[str, JsonValue]) -> Path:
    if _DB_PATH_CONFIG_KEY not in config:
        raise ValueError(
            f"sqlite-file qualification provider requires {_DB_PATH_CONFIG_KEY!r} config",
        )
    raw = config[_DB_PATH_CONFIG_KEY]
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            f"sqlite-file qualification provider {_DB_PATH_CONFIG_KEY!r} must be a non-empty str",
        )
    return Path(raw)


def sqlite_file_evidence_provider_factory(
    descriptor: EvidenceProviderDescriptor,
) -> EvidencePersistencePort:
    if descriptor.provider_id != SQLITE_FILE_PROVIDER_ID:
        raise ValueError(
            f"sqlite_file_evidence_provider_factory expected provider_id "
            f"{SQLITE_FILE_PROVIDER_ID!r}, got {descriptor.provider_id!r}",
        )
    db_path = _require_db_path(descriptor.config)
    store = SQLiteRuntimeEventStore(db_path=db_path)
    port = as_evidence_persistence_port(store)
    if port is None:
        raise RuntimeError("sqlite qualification provider failed to produce EvidencePersistencePort")
    return port
