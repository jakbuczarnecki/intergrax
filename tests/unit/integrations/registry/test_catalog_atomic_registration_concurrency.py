# © Artur Czarnecki. All rights reserved.

"""Deterministic concurrency proofs for live catalog registration atomicity."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationEntry,
    IntegrationStatus,
)
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import (
    catalog_snapshot,
    clear_catalog,
    list_slugs,
    register_integration,
)
from intergrax.integrations.registry.catalog_mutation import current_catalog_revision
from intergrax.integrations.registry.catalog_revision import compute_catalog_state_digest

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


def test_reg_conc_1_concurrent_same_slug_exactly_one_success() -> None:
    slug = "race-concurrent-slug"
    entry = _entry(slug)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    outcome_lock = threading.Lock()

    def _attempt() -> None:
        barrier.wait()
        try:
            register_integration(entry, override=False)
            with outcome_lock:
                outcomes.append("success")
        except ValueError as exc:
            with outcome_lock:
                outcomes.append(exc.args[0])

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(_attempt), pool.submit(_attempt))
        for future in futures:
            future.result()

    assert outcomes.count("success") == 1
    duplicate_messages = [item for item in outcomes if item != "success"]
    assert len(duplicate_messages) == 1
    assert "already registered" in duplicate_messages[0].lower()


def test_reg_conc_2_final_catalog_contains_single_logical_entry() -> None:
    slug = "race-concurrent-slug"
    entry = _entry(slug)
    barrier = threading.Barrier(2)

    def _attempt() -> None:
        barrier.wait()
        try:
            register_integration(entry, override=False)
        except ValueError:
            pass

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(_attempt), pool.submit(_attempt))
        for future in futures:
            future.result()

    assert list_slugs() == [slug]
    assert len(catalog_snapshot()) == 1


def test_reg_conc_3_generation_increases_exactly_once_from_initial() -> None:
    slug = "race-generation-slug"
    entry = _entry(slug)
    initial = current_catalog_revision()
    barrier = threading.Barrier(2)

    def _attempt() -> None:
        barrier.wait()
        try:
            register_integration(entry, override=False)
        except ValueError:
            pass

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(_attempt), pool.submit(_attempt))
        for future in futures:
            future.result()

    final = current_catalog_revision()
    assert final.generation == initial.generation + 1


def test_reg_conc_4_final_digest_matches_final_state() -> None:
    slug = "race-digest-slug"
    entry = _entry(slug)
    barrier = threading.Barrier(2)

    def _attempt() -> None:
        barrier.wait()
        try:
            register_integration(entry, override=False)
        except ValueError:
            pass

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(_attempt), pool.submit(_attempt))
        for future in futures:
            future.result()

    snapshot = catalog_snapshot()
    rev = current_catalog_revision()
    assert rev.state_digest == compute_catalog_state_digest(snapshot)


def test_reg_conc_optional_concurrent_distinct_slugs_both_succeed() -> None:
    alpha = _entry("concurrent-alpha")
    beta = _entry("concurrent-beta")
    initial = current_catalog_revision()
    barrier = threading.Barrier(2)

    def _register_alpha() -> None:
        barrier.wait()
        register_integration(alpha, override=False)

    def _register_beta() -> None:
        barrier.wait()
        register_integration(beta, override=False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(_register_alpha), pool.submit(_register_beta))
        for future in futures:
            future.result()

    final = current_catalog_revision()
    assert final.generation == initial.generation + 2
    assert set(list_slugs()) == {"concurrent-alpha", "concurrent-beta"}
