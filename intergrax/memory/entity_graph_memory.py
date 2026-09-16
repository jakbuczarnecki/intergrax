# © Artur Czarnecki. All rights reserved.

"""Legacy entity graph DTOs and migration-only facade (MEM-ENT-7 · MEM-ENT-11)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from intergrax.memory.contracts.entity_temporal_memory import EntityTemporalMemoryStore
from intergrax.utils.time_provider import SystemTimeProvider, TimeProvider

_LEGACY_TENANT_ID = "legacy-default"


class EntityGraphLegacyBypassError(RuntimeError):
    """Ungoverned legacy entity graph API is not a production memory boundary."""


def _reject_legacy_graph_access(operation: str) -> None:
    raise EntityGraphLegacyBypassError(
        f"EntityGraphMemoryStore.{operation} bypasses EntityTemporalMemoryCapability "
        "governance; use resolve_entity_temporal_memory_capability() in composition roots."
    )


@dataclass(frozen=True, slots=True)
class EntityNode:
    entity_id: str
    label: str
    entity_type: str = "person"
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EntityEdge:
    source_id: str
    target_id: str
    relation: str
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None


class EntityGraphMemoryStore:
    """
    Migration-only facade; not a governed production memory boundary (MEM-ENT-11).

    Use ``EntityTemporalMemoryCapability`` for reads and mutations.
    """

    def __init__(
        self,
        backend: EntityTemporalMemoryStore,
        *,
        tenant_id: str = _LEGACY_TENANT_ID,
        time_provider: TimeProvider | None = None,
    ) -> None:
        warnings.warn(
            "EntityGraphMemoryStore is migration-only; use EntityTemporalMemoryCapability",
            DeprecationWarning,
            stacklevel=2,
        )
        self._backend = backend
        self._tenant_id = tenant_id
        self._time_provider = time_provider or SystemTimeProvider()

    @property
    def entity_temporal_store(self) -> EntityTemporalMemoryStore:
        return self._backend

    def upsert_node(self, node: EntityNode) -> None:
        _reject_legacy_graph_access("upsert_node")

    def add_edge(self, edge: EntityEdge) -> None:
        _reject_legacy_graph_access("add_edge")

    def neighbors(
        self,
        entity_id: str,
        *,
        as_of: datetime | None = None,
    ) -> List[EntityNode]:
        _reject_legacy_graph_access("neighbors")

    def list_nodes(self) -> List[EntityNode]:
        _reject_legacy_graph_access("list_nodes")
