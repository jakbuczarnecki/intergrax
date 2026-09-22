# © Artur Czarnecki. All rights reserved.

"""Integration catalog revision, digest, and CAS primitives."""

from __future__ import annotations

import pytest

from intergrax.contracts.integration_catalog_revision import CatalogRevision
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import (
    clear_catalog,
    register_integration,
    unregister_integration,
)
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


def test_rev1_initial_revision_is_gen0_empty_digest() -> None:
    rev = current_catalog_revision()
    assert rev.generation == 0
    assert rev.state_digest == compute_catalog_state_digest({})


def test_rev2_register_material_change_bumps_generation() -> None:
    before = current_catalog_revision()
    register_integration(_entry("alpha"), override=True)
    after = current_catalog_revision()
    assert after.generation == before.generation + 1
    assert after.state_digest != before.state_digest


def test_rev3_identical_logical_override_does_not_bump() -> None:
    register_integration(_entry("alpha"), override=True)
    before = current_catalog_revision()
    register_integration(_entry("alpha"), override=True)
    assert current_catalog_revision() == before


def test_rev4_unregister_existing_bumps_generation() -> None:
    register_integration(_entry("alpha"), override=True)
    before = current_catalog_revision()
    unregister_integration("alpha")
    after = current_catalog_revision()
    assert after.generation == before.generation + 1


def test_rev5_unregister_missing_does_not_bump() -> None:
    before = current_catalog_revision()
    unregister_integration("missing-slug")
    assert current_catalog_revision() == before


def test_rev6_empty_after_mutation_does_not_reset_generation() -> None:
    register_integration(_entry("only"), override=True)
    unregister_integration("only")
    rev = current_catalog_revision()
    assert rev.generation >= 1
    assert rev.state_digest == compute_catalog_state_digest({})


def test_rev7_aba_generations_increase_when_digest_returns() -> None:
    register_integration(_entry("alpha"), override=True)
    rev_a = current_catalog_revision()
    register_integration(_entry("beta"), override=True)
    rev_b = current_catalog_revision()
    unregister_integration("beta")
    rev_a_again = current_catalog_revision()
    assert rev_b.generation == rev_a.generation + 1
    assert rev_a_again.generation == rev_b.generation + 1
    assert rev_a_again.state_digest == rev_a.state_digest
    assert rev_a_again.generation != rev_a.generation


def test_rev8_same_digest_different_generation_after_aba() -> None:
    register_integration(_entry("alpha"), override=True)
    first = current_catalog_revision()
    register_integration(_entry("beta"), override=True)
    unregister_integration("beta")
    second = current_catalog_revision()
    assert second.state_digest == first.state_digest
    assert second.generation > first.generation


def test_aba_stale_authorization_revision_conflict() -> None:
    register_integration(_entry("alpha"), override=True)
    authorized = current_catalog_revision()
    register_integration(_entry("beta"), override=True)
    unregister_integration("beta")
    result = replace_catalog_if_revision(
        expected_revision=authorized,
        candidate_entries={"alpha": _entry("alpha")},
    )
    assert result.outcome is CatalogReplaceOutcome.REVISION_CONFLICT
