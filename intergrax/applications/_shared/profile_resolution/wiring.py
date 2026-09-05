# © Artur Czarnecki. All rights reserved.

"""Composition wiring for effective profile revision persistence (P1.2A)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.profile_resolution.durability_policy import (
    validate_effective_profile_pinning_durability_for_composition,
)
from intergrax.applications._shared.profile_resolution.execution_pinning import (
    InMemoryEffectiveProfileExecutionPinningStore,
)
from intergrax.applications._shared.profile_resolution.persistence import (
    wire_effective_profile_execution_pinning_store,
    wire_effective_profile_revision_store,
)
from intergrax.applications._shared.profile_resolution.store import (
    InMemoryEffectiveProfileRevisionStore,
)
from intergrax.applications.contracts.profile_resolution.execution_binding import (
    EffectiveProfileExecutionPinningStore,
)
from intergrax.applications.contracts.profile_resolution.store import (
    EffectiveProfileRevisionStore,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import DocumentStore


@dataclass(frozen=True, slots=True)
class EffectiveProfilePersistenceWiring:
    """Resolved revision and pinning stores for one host composition."""

    revision_store: EffectiveProfileRevisionStore
    pinning_store: EffectiveProfileExecutionPinningStore


def resolve_effective_profile_persistence_wiring(
    *,
    production_mode: bool,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
    revision_store: EffectiveProfileRevisionStore | None = None,
    pinning_store: EffectiveProfileExecutionPinningStore | None = None,
) -> EffectiveProfilePersistenceWiring:
    """Resolve revision/pinning stores from explicit adapters or platform primitives."""
    if revision_store is not None and pinning_store is not None:
        resolved_revision_store = revision_store
        resolved_pinning_store = pinning_store
    elif kv_store is not None or document_store is not None:
        if revision_store is not None:
            resolved_revision_store = revision_store
        else:
            resolved_revision_store = wire_effective_profile_revision_store(
                kv_store=kv_store,
                document_store=document_store,
            )
        if pinning_store is not None:
            resolved_pinning_store = pinning_store
        else:
            resolved_pinning_store = wire_effective_profile_execution_pinning_store(
                kv_store=kv_store,
                document_store=document_store,
            )
    else:
        resolved_revision_store = revision_store or InMemoryEffectiveProfileRevisionStore()
        resolved_pinning_store = pinning_store or InMemoryEffectiveProfileExecutionPinningStore()

    validate_effective_profile_pinning_durability_for_composition(
        production_mode=production_mode,
        revision_store=resolved_revision_store,
        pinning_store=resolved_pinning_store,
    )
    return EffectiveProfilePersistenceWiring(
        revision_store=resolved_revision_store,
        pinning_store=resolved_pinning_store,
    )
