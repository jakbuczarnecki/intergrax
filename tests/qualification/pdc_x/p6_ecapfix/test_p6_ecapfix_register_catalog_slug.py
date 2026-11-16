# © Artur Czarnecki. All rights reserved.

"""T-IGX-036-X (B-P6-F): the single ``ollama`` catalog row resolves both of its categories.

Since B-REG-P the ``ollama`` embedding provider augments the model-serving row instead of
registering a second row. Two layers are pinned here: the ebh_5 package C v2 harness must
register the owning model-serving row first, and the factory of the single row must dispatch on
``integration_category``. The negative test calls the product registration function directly:
the product contract (augment requires the owning row) is unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from intergrax.integrations.contracts.base import IntegrationCategory, UnknownIntegrationError
from intergrax.integrations.providers.embedding_provider.ollama.register import (
    register_ollama_embedding_provider_integration,
)
from intergrax.integrations.providers.model_serving_runtime.ollama.integration import (
    OllamaModelServingRuntimeIntegration,
)
from intergrax.integrations.providers.model_serving_runtime.ollama.register import register_ollama_integration
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
from intergrax.integrations.registry.factory import resolve_from_profile
from intergrax.runtime.integrations.categories.embedding import EmbeddingProviderIntegrationContract
from tests.qualification.ebh_5._package_c_v2_primitives import profile_for_slug, register_catalog_slug

pytestmark = pytest.mark.unit

_SLUG = "ollama"


@pytest.fixture()
def cleared_catalog() -> Iterator[None]:
    """Empty catalog for the test; restore the previous process state afterwards."""
    with catalog_state_lock():
        saved_entries = catalog_snapshot()
        saved_generation = read_catalog_revision_under_lock().generation
    clear_catalog()
    try:
        yield
    finally:
        clear_catalog()
        for entry in saved_entries.values():
            register_integration(entry, override=True)
        set_catalog_generation_for_tests(saved_generation)


def test_harness_registers_ollama_and_hf_embedding_providers_on_cleared_catalog(cleared_catalog: None) -> None:
    register_catalog_slug(category=IntegrationCategory.EMBEDDING_PROVIDER, slug=_SLUG)
    register_catalog_slug(category=IntegrationCategory.EMBEDDING_PROVIDER, slug="hf")

    slugs = list_slugs(category=IntegrationCategory.EMBEDDING_PROVIDER)
    assert _SLUG in slugs
    assert "hf" in slugs
    assert get_entry(_SLUG).slug == _SLUG
    assert get_entry("hf").slug == "hf"


def test_harness_leaves_one_ollama_row_with_both_contributions(cleared_catalog: None) -> None:
    register_catalog_slug(category=IntegrationCategory.EMBEDDING_PROVIDER, slug=_SLUG)

    assert list_slugs().count(_SLUG) == 1
    entry = get_entry(_SLUG)
    assert IntegrationCategory.MODEL_SERVING_RUNTIME in entry.categories
    assert IntegrationCategory.EMBEDDING_PROVIDER in entry.categories
    spec_categories = {str(spec.category) for spec in entry.contract_specs}
    assert "model_serving_runtime" in spec_categories
    assert "embedding_provider" in spec_categories
    assert len(entry.contract_specs) == len(spec_categories)


def test_product_embedding_registration_without_owning_row_still_fails_closed(cleared_catalog: None) -> None:
    with pytest.raises(UnknownIntegrationError):
        register_ollama_embedding_provider_integration()

    assert _SLUG not in list_slugs()


def _register_owning_row_then_embedding_contribution() -> None:
    register_ollama_integration()
    register_ollama_embedding_provider_integration()


def test_single_ollama_row_carries_both_categories_and_contract_specs(cleared_catalog: None) -> None:
    _register_owning_row_then_embedding_contribution()

    assert list_slugs().count(_SLUG) == 1
    entry = get_entry(_SLUG)
    assert set(entry.categories) == {
        IntegrationCategory.MODEL_SERVING_RUNTIME,
        IntegrationCategory.EMBEDDING_PROVIDER,
    }
    assert {str(spec.category) for spec in entry.contract_specs} == {"model_serving_runtime", "embedding_provider"}


def test_resolve_embedding_category_returns_embedding_provider_integration(cleared_catalog: None) -> None:
    _register_owning_row_then_embedding_contribution()
    category = IntegrationCategory.EMBEDDING_PROVIDER

    resolved = resolve_from_profile(profile_for_slug(category=category, slug=_SLUG), category)

    assert isinstance(resolved, EmbeddingProviderIntegrationContract)
    assert not isinstance(resolved, OllamaModelServingRuntimeIntegration)


def test_resolve_model_serving_category_is_unchanged(cleared_catalog: None) -> None:
    _register_owning_row_then_embedding_contribution()
    category = IntegrationCategory.MODEL_SERVING_RUNTIME

    resolved = resolve_from_profile(profile_for_slug(category=category, slug=_SLUG), category)

    assert isinstance(resolved, OllamaModelServingRuntimeIntegration)
    assert not isinstance(resolved, EmbeddingProviderIntegrationContract)


def test_model_serving_row_alone_still_resolves_model_serving_without_embedding_contribution(
    cleared_catalog: None,
) -> None:
    register_ollama_integration()
    category = IntegrationCategory.MODEL_SERVING_RUNTIME

    resolved = resolve_from_profile(profile_for_slug(category=category, slug=_SLUG), category)

    assert isinstance(resolved, OllamaModelServingRuntimeIntegration)
