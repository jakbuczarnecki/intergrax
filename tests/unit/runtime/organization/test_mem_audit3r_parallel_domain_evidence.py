# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-3-R: Organization Memory parallel-domain evidence proofs."""

from __future__ import annotations

import pytest

from intergrax.runtime.organization.organization_profile import OrganizationProfile
from intergrax.runtime.organization.organization_profile_manager import OrganizationProfileManager
from intergrax.runtime.organization.organization_profile_store import OrganizationProfileStore
from intergrax.runtime.organization.stores.in_memory_organization_profile_store import (
    InMemoryOrganizationProfileStore,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class FailingOrganizationProfileStore(InMemoryOrganizationProfileStore):
    """Contract-based fault injection for OrganizationProfileStore."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_operation: str = ""
        self.fail_organization_id: str | None = None

    async def save_profile(self, profile: OrganizationProfile) -> None:
        org_id = profile.identity.organization_id
        if self.fail_operation == "save" and (
            self.fail_organization_id is None or org_id == self.fail_organization_id
        ):
            raise RuntimeError("organization save failed")
        await super().save_profile(profile)

    async def get_profile(self, organization_id: str) -> OrganizationProfile:
        if self.fail_operation == "get":
            raise RuntimeError("organization get failed")
        return await super().get_profile(organization_id)

    async def delete_profile(self, organization_id: str) -> None:
        if self.fail_operation == "delete":
            raise RuntimeError("organization delete failed")
        await super().delete_profile(organization_id)


@pytest.mark.asyncio
async def test_in_memory_organization_store_documents_caller_serialized_concurrency() -> None:
    doc = InMemoryOrganizationProfileStore.__doc__ or ""
    lowered = doc.lower()
    assert "not thread-safe" in lowered
    assert "caller" in lowered


@pytest.mark.asyncio
async def test_organization_profiles_isolated_by_organization_id() -> None:
    store = InMemoryOrganizationProfileStore()
    profile_a = await store.get_profile("org-a")
    profile_a.system_instructions = "org-a-only"
    await store.save_profile(profile_a)

    profile_b = await store.get_profile("org-b")
    assert profile_b.system_instructions != "org-a-only"


@pytest.mark.asyncio
async def test_organization_save_failure_does_not_corrupt_other_org() -> None:
    store = FailingOrganizationProfileStore()
    store.fail_operation = "save"
    store.fail_organization_id = "org-bad"
    good = await store.get_profile("org-good")
    good.system_instructions = "stable"
    await store.save_profile(good)

    bad = await store.get_profile("org-bad")
    bad.system_instructions = "attempt"
    with pytest.raises(RuntimeError, match="save failed"):
        await store.save_profile(bad)

    reloaded = await store.get_profile("org-good")
    assert reloaded.system_instructions == "stable"


@pytest.mark.asyncio
async def test_organization_get_failure_surfaces_to_manager() -> None:
    store = FailingOrganizationProfileStore()
    store.fail_operation = "get"
    manager = OrganizationProfileManager(store)
    with pytest.raises(RuntimeError, match="get failed"):
        await manager.get_profile("org-x")


@pytest.mark.asyncio
async def test_organization_delete_failure_surfaces_to_caller() -> None:
    store = FailingOrganizationProfileStore()
    profile = await store.get_profile("org-x")
    profile.system_instructions = "keep"
    await store.save_profile(profile)
    store.fail_operation = "delete"
    with pytest.raises(RuntimeError, match="delete failed"):
        await store.delete_profile("org-x")
    reloaded = await store.get_profile("org-x")
    assert reloaded.system_instructions == "keep"


@pytest.mark.asyncio
async def test_organization_profile_store_protocol_documents_concurrency() -> None:
    doc = OrganizationProfileStore.__doc__ or ""
    assert "concurrency" in doc.lower()


def test_organization_revision_semantics_not_exposed_on_store_contract() -> None:
    """Stale optimistic revision is N/A — store uses overwrite semantics only."""
    import inspect

    source = inspect.getsource(InMemoryOrganizationProfileStore.save_profile)
    assert "revision" not in source.lower()
    assert "expected_version" not in source.lower()
