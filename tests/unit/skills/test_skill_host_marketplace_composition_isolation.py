# © Artur Czarnecki. All rights reserved.

"""EC-8 evidence: per-host lifecycle/registry isolation in marketplace composition."""

from __future__ import annotations

import pytest

from intergrax.skills.host_lifecycle import SkillHostLifecycleService
from testing_support.canonical_me15_reference_skill import ME15_SKILL_LOGICAL_ID
from testing_support.marketplace_skill_composition import MarketplaceSkillCompositionProofStack
from tests.unit.skills.test_skill_host_lifecycle_bind_failure_paths import (
    _MutateThenRaiseMaterializer,
    _resolution,
)
from testing_support.me15_skill_catalog_provider import Me15SkillCatalogProvider

pytestmark = pytest.mark.unit


def test_marketplace_composition_allocates_distinct_host_registries() -> None:
    shared_provider = Me15SkillCatalogProvider()
    stack_a = MarketplaceSkillCompositionProofStack.build(
        catalog_provider=shared_provider,
        lifecycle=SkillHostLifecycleService(host_profile_id="tenant-a-host"),
    )
    stack_b = MarketplaceSkillCompositionProofStack.build(
        catalog_provider=shared_provider,
        lifecycle=SkillHostLifecycleService(host_profile_id="tenant-b-host"),
    )
    assert stack_a.lifecycle.registry is not stack_b.lifecycle.registry
    assert stack_a.lifecycle.skill_profile is not stack_b.lifecycle.skill_profile
    assert stack_a.acquisition is not stack_b.acquisition
    assert stack_a.lifecycle.host_profile_id != stack_b.lifecycle.host_profile_id


def test_failed_bind_on_one_host_does_not_mutate_peer_marketplace_stack() -> None:
    shared_provider = Me15SkillCatalogProvider()
    stack_a = MarketplaceSkillCompositionProofStack.build(
        catalog_provider=shared_provider,
        lifecycle=SkillHostLifecycleService(host_profile_id="tenant-a-fail"),
    )
    stack_b = MarketplaceSkillCompositionProofStack.build(
        catalog_provider=shared_provider,
        lifecycle=SkillHostLifecycleService(host_profile_id="tenant-b-safe"),
    )
    resolution = _resolution(shared_provider)
    stack_a.lifecycle.bind(
        operation_id="op-a-fail",
        host_profile_id=stack_a.lifecycle.host_profile_id,
        resolved=resolution,
        materializer=_MutateThenRaiseMaterializer(
            stack_a.lifecycle.registry,
            catalog_source_id=shared_provider.catalog_source_id,
        ),
    )
    assert not stack_a.lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)
    assert not stack_b.lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)
