# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-15-R2: production entity projection composition."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.entity_user_profile_memory_projection import (
    EntityIndexerUserProfileMemoryProjection,
)
from intergrax.applications._shared.memory_vector_wiring import build_user_profile_manager
from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore

pytestmark = pytest.mark.gate


def test_build_user_profile_manager_includes_entity_projection_when_enabled() -> None:
    env = ApplicationEnvironmentProfile(
        memory_profile=MemoryProfile(
            enable_user_memory=True,
            enable_entity_graph_memory=True,
        ),
    )
    wiring = resolve_memory_platform_wiring(env, tenant_id="tenant-wiring")
    manager = build_user_profile_manager(
        InMemoryUserProfileStore(),
        env,
        tenant_id="tenant-wiring",
        entity_memory_indexer=wiring.entity_memory_indexer,
        entity_temporal_memory_capability=wiring.entity_temporal_memory_capability,
    )
    assert manager is not None
    projections = manager._memory_lifecycle.projections
    assert any(isinstance(p, EntityIndexerUserProfileMemoryProjection) for p in projections)


def test_custom_entity_indexer_injected_via_projection_adapter() -> None:
    indexer = MagicMock()
    capability = MagicMock()
    adapter = EntityIndexerUserProfileMemoryProjection(
        indexer=indexer,
        entity_capability=capability,
    )
    assert adapter.indexer is indexer
