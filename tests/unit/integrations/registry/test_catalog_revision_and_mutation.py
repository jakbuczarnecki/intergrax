# © Artur Czarnecki. All rights reserved.

"""Integration catalog revision, digest, and CAS primitives."""

from __future__ import annotations

import pytest

from intergrax.contracts.integration_catalog_revision import CatalogRevision
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog, register_integration
from intergrax.integrations.registry.catalog_mutation import (
    CatalogReplaceOutcome,
    build_catalog_entries_for_preset,
    current_catalog_revision,
    replace_catalog_if_revision,
)
from intergrax.integrations.registry.catalog_revision import compute_catalog_state_digest
from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationEntry,
    IntegrationStatus,
)

pytestmark = pytest.mark.unit


def _entry(slug: str) -> IntegrationEntry:
    return IntegrationEntry(
        slug=slug,
        categories=(IntegrationCategory.KEY_VALUE_CACHE,),
        factory=lambda: None,
        status=IntegrationStatus.STABLE,
    )


@pytest.fixture(autouse=True)
def _reset_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()


def test_digest_stable_for_same_logical_state() -> None:
    a = {"redis": _entry("redis")}
    b = {"redis": _entry("redis")}
    assert compute_catalog_state_digest(a) == compute_catalog_state_digest(b)


def test_replace_catalog_cas_conflict() -> None:
    register_integration(_entry("alpha"), override=True)
    before = current_catalog_revision()
    stale = CatalogRevision(generation=before.generation, state_digest="0" * 64)
    result = replace_catalog_if_revision(
        expected_revision=stale,
        candidate_entries=build_catalog_entries_for_preset("core"),
    )
    assert result.outcome is CatalogReplaceOutcome.REVISION_CONFLICT
    assert current_catalog_revision() == before


def test_build_preset_candidate_does_not_mutate_live_catalog() -> None:
    register_integration(_entry("live"), override=True)
    before = current_catalog_revision()
    build_catalog_entries_for_preset("core")
    assert current_catalog_revision() == before
