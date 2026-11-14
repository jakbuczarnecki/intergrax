# © Artur Czarnecki. All rights reserved.

"""T-IGX-040-A (B-P6-J, F-49): the single ``openai`` catalog row resolves both of its categories.

``managed_retrieval/openai`` owns the ``openai`` slug; the embedding provider only augments that row.
The factory of the row must dispatch on ``integration_category`` like the ``ollama`` row (T-IGX-036-X):
the embedding category builds the embedding provider integration, every other category keeps the owning
factory. No network, no client and no real credential is involved (a placeholder value is set where the
managed-retrieval path needs a non-empty key).
"""

from __future__ import annotations

import pytest
from intergrax.integrations.contracts.base import IntegrationCategory, UnknownIntegrationError
from intergrax.integrations.providers.embedding_provider.openai.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.embedding_provider.openai.integration import (
    OpenaiEmbeddingProviderIntegration,
)
from intergrax.integrations.providers.embedding_provider.openai.register import (
    register_openai_embedding_provider_integration,
)
from intergrax.integrations.providers.managed_retrieval.openai.bundle import (
    create_openai_managed_retrieval_integration,
)
from intergrax.integrations.providers.managed_retrieval.openai.integration import (
    OpenAIManagedRetrievalIntegration,
)
from intergrax.integrations.providers.managed_retrieval.openai.register import (
    register_openai_managed_retrieval_integration,
)
from intergrax.integrations.registry.catalog import (
    augment_integration_contract_specs,
    catalog_snapshot,
    clear_catalog,
    compute_catalog_state_digest,
    get_entry,
    list_slugs,
    read_catalog_revision_under_lock,
)
from intergrax.integrations.registry.factory import resolve
from intergrax.runtime.integrations.categories.embedding import EmbeddingProviderIntegrationContract
from tests.qualification.pdc_x.p6_ecapfix._wrapper_support import (
    RecordingFactory,
    install_recording_owning_factory,
    wrapper_depth,
)

pytestmark = pytest.mark.unit

_SLUG = "openai"
_REGISTER_MODULE = "intergrax.integrations.providers.embedding_provider.openai.register"
_PLACEHOLDER_VALUE = "placeholder-not-a-credential"


@pytest.fixture()
def no_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _register_owning_row_then_embedding_contribution() -> None:
    register_openai_managed_retrieval_integration()
    register_openai_embedding_provider_integration()


def test_single_openai_row_carries_both_categories_and_contract_specs(cleared_catalog: None) -> None:
    _register_owning_row_then_embedding_contribution()

    assert list_slugs().count(_SLUG) == 1
    entry = get_entry(_SLUG)
    assert set(entry.categories) == {
        IntegrationCategory.MANAGED_RETRIEVAL,
        IntegrationCategory.EMBEDDING_PROVIDER,
    }
    assert {str(spec.category) for spec in entry.contract_specs} == {"managed_retrieval", "embedding_provider"}


def test_resolve_embedding_category_without_api_key_returns_embedding_provider_integration(
    cleared_catalog: None, no_openai_key: None
) -> None:
    _register_owning_row_then_embedding_contribution()

    resolved = resolve(IntegrationCategory.EMBEDDING_PROVIDER, _SLUG)

    assert isinstance(resolved, EmbeddingProviderIntegrationContract)
    assert isinstance(resolved, OpenaiEmbeddingProviderIntegration)
    assert not isinstance(resolved, OpenAIManagedRetrievalIntegration)


def test_resolve_embedding_category_with_environment_value_still_returns_embedding_provider_integration(
    cleared_catalog: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", _PLACEHOLDER_VALUE)
    _register_owning_row_then_embedding_contribution()

    resolved = resolve(IntegrationCategory.EMBEDDING_PROVIDER, _SLUG)

    assert isinstance(resolved, OpenaiEmbeddingProviderIntegration)


def test_resolve_managed_retrieval_category_is_unchanged(
    cleared_catalog: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", _PLACEHOLDER_VALUE)
    _register_owning_row_then_embedding_contribution()

    resolved = resolve(IntegrationCategory.MANAGED_RETRIEVAL, _SLUG)

    assert isinstance(resolved, OpenAIManagedRetrievalIntegration)
    assert not isinstance(resolved, EmbeddingProviderIntegrationContract)


def test_managed_retrieval_without_api_key_still_fails_closed(cleared_catalog: None, no_openai_key: None) -> None:
    _register_owning_row_then_embedding_contribution()

    with pytest.raises(Exception, match="OPENAI_API_KEY"):
        resolve(IntegrationCategory.MANAGED_RETRIEVAL, _SLUG)


def test_embedding_registration_without_owning_row_still_fails_closed(cleared_catalog: None) -> None:
    with pytest.raises(UnknownIntegrationError):
        register_openai_embedding_provider_integration()

    assert _SLUG not in list_slugs()


def test_enabled_defaults_to_false_for_the_embedding_category(cleared_catalog: None, no_openai_key: None) -> None:
    _register_owning_row_then_embedding_contribution()

    resolved = resolve(IntegrationCategory.EMBEDDING_PROVIDER, _SLUG)

    assert isinstance(resolved, OpenaiEmbeddingProviderIntegration)
    assert resolved.config.enabled is False


def test_enabled_true_reaches_the_embedding_integration(cleared_catalog: None, no_openai_key: None) -> None:
    _register_owning_row_then_embedding_contribution()

    resolved = resolve(IntegrationCategory.EMBEDDING_PROVIDER, _SLUG, config={"enabled": True})

    assert isinstance(resolved, OpenaiEmbeddingProviderIntegration)
    assert resolved.config.enabled is True


def test_embedding_category_does_not_call_the_owning_factory(cleared_catalog: None, no_openai_key: None) -> None:
    register_openai_managed_retrieval_integration()
    recorder = RecordingFactory(create_openai_managed_retrieval_integration)
    install_recording_owning_factory(_SLUG, recorder)
    register_openai_embedding_provider_integration()

    resolved = resolve(IntegrationCategory.EMBEDDING_PROVIDER, _SLUG)

    assert isinstance(resolved, OpenaiEmbeddingProviderIntegration)
    assert recorder.calls == []


def test_non_embedding_keyword_arguments_reach_the_owning_factory_unchanged(cleared_catalog: None) -> None:
    register_openai_managed_retrieval_integration()
    recorder = RecordingFactory(create_openai_managed_retrieval_integration)
    install_recording_owning_factory(_SLUG, recorder)
    register_openai_embedding_provider_integration()

    resolved = resolve(IntegrationCategory.MANAGED_RETRIEVAL, _SLUG, config={"enabled": False, "marker": 7})

    assert isinstance(resolved, OpenAIManagedRetrievalIntegration)
    assert recorder.calls == [
        {"integration_category": IntegrationCategory.MANAGED_RETRIEVAL, "enabled": False, "marker": 7}
    ]


def test_catalog_digest_and_revision_equal_the_augment_only_state(cleared_catalog: None) -> None:
    register_openai_managed_retrieval_integration()
    augment_integration_contract_specs(
        _SLUG,
        categories=(IntegrationCategory.EMBEDDING_PROVIDER,),
        contract_specs=CONTRACT_SPECS,
    )
    augment_only_digest = compute_catalog_state_digest(catalog_snapshot())
    augment_only_revision = read_catalog_revision_under_lock()

    clear_catalog()
    _register_owning_row_then_embedding_contribution()

    assert compute_catalog_state_digest(catalog_snapshot()) == augment_only_digest
    assert read_catalog_revision_under_lock().state_digest == augment_only_revision.state_digest
    assert list_slugs().count(_SLUG) == 1


def test_second_embedding_registration_raises_and_does_not_nest_the_wrapper(cleared_catalog: None) -> None:
    _register_owning_row_then_embedding_contribution()
    assert wrapper_depth(get_entry(_SLUG).factory, module_name=_REGISTER_MODULE) == 1

    with pytest.raises(ValueError, match="already registered"):
        register_openai_embedding_provider_integration()

    assert wrapper_depth(get_entry(_SLUG).factory, module_name=_REGISTER_MODULE) == 1


def test_rebootstrap_with_override_leaves_exactly_one_wrapper_level(cleared_catalog: None) -> None:
    _register_owning_row_then_embedding_contribution()

    register_openai_managed_retrieval_integration(override=True)
    register_openai_embedding_provider_integration(override=True)

    assert list_slugs().count(_SLUG) == 1
    assert wrapper_depth(get_entry(_SLUG).factory, module_name=_REGISTER_MODULE) == 1
