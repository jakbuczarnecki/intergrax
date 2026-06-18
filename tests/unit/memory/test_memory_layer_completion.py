# © Artur Czarnecki. All rights reserved.

"""Layer completion tests for MEMORY platform hardening."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from intergrax.memory.memory_temporal import is_memory_entry_active
from intergrax.memory.memory_vector_namespace import resolve_memory_index_collection
from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry
from intergrax.runtime.organization.organization_profile import OrganizationIdentity, OrganizationProfile
from intergrax.runtime.organization.organization_profile_manager import OrganizationProfileManager
from intergrax.runtime.organization.stores.in_memory_organization_profile_store import (
    InMemoryOrganizationProfileStore,
)
from intergrax.runtime.task_memory.metrics import memory_platform_metrics, reset_memory_platform_metrics_for_tests

pytestmark = pytest.mark.unit


def test_resolve_memory_index_collection_defaults_to_tenant_domain() -> None:
    assert resolve_memory_index_collection(
        vector_index_namespace=None,
        tenant_id="tenant-a",
        domain="ltm",
    ) == "tenant-a:ltm"


def test_is_memory_entry_active_respects_valid_until() -> None:
    now = datetime(2026, 6, 17, tzinfo=timezone.utc)
    entry = UserProfileMemoryEntry(
        content="fact",
        valid_until=(now - timedelta(days=1)).isoformat(),
    )
    assert is_memory_entry_active(entry, as_of=now) is False


@pytest.mark.asyncio
async def test_organization_profile_manager_memory_entries() -> None:
    manager = OrganizationProfileManager(InMemoryOrganizationProfileStore())
    entry = UserProfileMemoryEntry(content="Org prefers markdown", kind=MemoryKind.ORG_FACT)
    stored = await manager.add_memory_entry("org-1", entry)
    assert stored.entry_id
    hits = await manager.search_memory_entries("org-1", "markdown")
    assert len(hits) == 1


def test_memory_platform_metrics_record_ltm_and_episodic() -> None:
    reset_memory_platform_metrics_for_tests()
    metrics = memory_platform_metrics()
    metrics.record_ltm_hit()
    metrics.record_episodic_hit()
    lines = metrics.prometheus_lines()
    assert "intergrax_memory_ltm_hits_total 1" in lines
    assert "intergrax_memory_episodic_hits_total 1" in lines
