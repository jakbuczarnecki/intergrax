# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-15-R2/R3: production entity projection composition (public behavioral proof)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.applications._shared.memory_vector_wiring import build_user_profile_manager
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.memory.contracts.entity_temporal_memory import EntityMemoryScope
from intergrax.memory.entity_temporal_memory_service import EntityTemporalMemoryService
from intergrax.memory.memory_security_governance_service import (
    build_default_memory_security_governance_service,
)
from intergrax.memory.stores.in_memory_entity_temporal_memory_store import (
    InMemoryEntityTemporalMemoryStore,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry

pytestmark = pytest.mark.gate


@dataclass(slots=True)
class RecordingEntityMemoryIndexer:
    """Typed fake implementing ``EntityMemoryIndexer`` for composition certification."""

    indexed: list[tuple[RequestIdentity, EntityMemoryScope, UserProfileMemoryEntry]] = field(
        default_factory=list,
    )
    removed: list[tuple[RequestIdentity, EntityMemoryScope, str]] = field(default_factory=list)

    def index_memory_entry(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        entry: UserProfileMemoryEntry,
    ) -> None:
        self.indexed.append((identity, scope, entry))

    def remove_memory_entry(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        memory_entry_id: str,
    ) -> None:
        self.removed.append((identity, scope, memory_entry_id))


def _entity_capability() -> EntityTemporalMemoryService:
    return EntityTemporalMemoryService(
        _store=InMemoryEntityTemporalMemoryStore(),
        _security_governance=build_default_memory_security_governance_service(),
    )


def _enabled_env() -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile(
        memory_profile=MemoryProfile(
            enable_user_memory=True,
            enable_entity_graph_memory=True,
        ),
    )


@pytest.mark.asyncio
async def test_production_builder_wires_entity_indexer_on_memory_mutation() -> None:
    recording = RecordingEntityMemoryIndexer()
    manager = build_user_profile_manager(
        InMemoryUserProfileStore(),
        _enabled_env(),
        tenant_id="tenant-r3",
        entity_memory_indexer=recording,
        entity_temporal_memory_capability=_entity_capability(),
    )
    assert manager is not None

    identity = RequestIdentity(
        tenant_id="tenant-r3",
        user_id="user-r3",
        principal_type=PrincipalType.SERVICE,
        auth_subject="mem-ent15-r3",
    )
    stored = await manager.add_memory_entry(identity, "user-r3", "entity wiring proof")

    assert len(recording.indexed) == 1
    observed_identity, observed_scope, observed_entry = recording.indexed[0]
    assert observed_identity == identity
    assert observed_scope.tenant_id == identity.tenant_id
    assert observed_scope.user_id == identity.user_id
    assert observed_scope.workspace_id is None
    assert observed_entry.entry_id == stored.entry_id


@pytest.mark.asyncio
async def test_entity_indexer_not_called_when_entity_graph_memory_disabled() -> None:
    recording = RecordingEntityMemoryIndexer()
    env = ApplicationEnvironmentProfile(
        memory_profile=MemoryProfile(
            enable_user_memory=True,
            enable_entity_graph_memory=False,
        ),
    )
    manager = build_user_profile_manager(
        InMemoryUserProfileStore(),
        env,
        tenant_id="tenant-r3-off",
        entity_memory_indexer=recording,
        entity_temporal_memory_capability=_entity_capability(),
    )
    assert manager is not None

    identity = RequestIdentity(
        tenant_id="tenant-r3-off",
        user_id="user-r3-off",
        principal_type=PrincipalType.SERVICE,
        auth_subject="mem-ent15-r3-off",
    )
    await manager.add_memory_entry(identity, "user-r3-off", "no entity projection")

    assert recording.indexed == []
    assert recording.removed == []
