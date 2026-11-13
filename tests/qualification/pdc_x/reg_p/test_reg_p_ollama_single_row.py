# © Artur Czarnecki. All rights reserved.

"""PDC-D1-R022 (B-REG-P): one ``ollama`` catalog row carries both contributions.

Regression from ``7aa9cfa18``: the embedding provider registered a second
catalog row with slug ``ollama``, so ``register_default_integrations(preset="full")``
raised ``ValueError: Integration slug 'ollama' is already registered`` on a fresh
catalog and every default ``create_default_embedding_manager()`` failed.
"""

from __future__ import annotations

from collections.abc import Iterator

import intergrax.integrations.registry.bootstrap as bootstrap_module
import pytest
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import (
    register_default_integrations,
    reset_default_integrations_state,
)
from intergrax.integrations.registry.catalog import (
    catalog_snapshot,
    catalog_state_lock,
    clear_catalog,
    get_entry,
    list_slugs,
    read_catalog_revision_under_lock,
    register_integration,
    set_catalog_generation_for_tests,
)

pytestmark = pytest.mark.unit

_SLUG = "ollama"


@pytest.fixture()
def fresh_catalog() -> Iterator[None]:
    """Empty catalog and bootstrap flag; restore the previous process state afterwards."""
    with catalog_state_lock():
        saved_entries = catalog_snapshot()
        saved_generation = read_catalog_revision_under_lock().generation
    saved_bootstrapped = bootstrap_module._BOOTSTRAPPED
    clear_catalog()
    reset_default_integrations_state()
    try:
        yield
    finally:
        clear_catalog()
        for entry in saved_entries.values():
            register_integration(entry, override=True)
        set_catalog_generation_for_tests(saved_generation)
        bootstrap_module._BOOTSTRAPPED = saved_bootstrapped


def test_full_preset_registers_on_fresh_catalog_without_duplicate_slug(fresh_catalog: None) -> None:
    register_default_integrations(preset="full")

    slugs = list_slugs()
    assert slugs.count(_SLUG) == 1
    assert len(slugs) == len(set(slugs))


def test_single_ollama_row_carries_model_serving_and_embedding_contribution(fresh_catalog: None) -> None:
    register_default_integrations(preset="full")

    entry = get_entry(_SLUG)
    assert IntegrationCategory.MODEL_SERVING_RUNTIME in entry.categories
    assert IntegrationCategory.EMBEDDING_PROVIDER in entry.categories
    spec_categories = {str(spec.category) for spec in entry.contract_specs}
    assert "model_serving_runtime" in spec_categories
    assert "embedding_provider" in spec_categories
    assert len(entry.contract_specs) == len(spec_categories)
    assert _SLUG in list_slugs(category=IntegrationCategory.MODEL_SERVING_RUNTIME)
    assert _SLUG in list_slugs(category=IntegrationCategory.EMBEDDING_PROVIDER)


def test_default_embedding_manager_builds_on_fresh_catalog(fresh_catalog: None) -> None:
    from intergrax.rag.embedding.bootstrap.default_embedding_engine import (
        create_default_embedding_manager,
    )

    manager = create_default_embedding_manager()

    assert manager is not None
