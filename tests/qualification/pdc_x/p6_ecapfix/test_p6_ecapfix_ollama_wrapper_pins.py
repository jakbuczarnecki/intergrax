# © Artur Czarnecki. All rights reserved.

"""T-IGX-040-A (B-P6-J, F-50): test-only pins for the ``ollama`` dispatching wrapper (T-IGX-036-X).

The wrapper itself is unchanged. These tests pin the behaviour that mutation sampling showed unpinned:
``enabled`` default and pass-through for the embedding category, and unchanged keyword arguments for
every other category. They also pin that repeated registration does not nest the wrapper.
"""

from __future__ import annotations

import pytest
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.embedding_provider.ollama.integration import (
    OllamaEmbeddingProviderIntegration,
)
from intergrax.integrations.providers.embedding_provider.ollama.register import (
    register_ollama_embedding_provider_integration,
)
from intergrax.integrations.providers.model_serving_runtime.ollama.integration import (
    OllamaModelServingRuntimeIntegration,
)
from intergrax.integrations.providers.model_serving_runtime.ollama.register import register_ollama_integration
from intergrax.integrations.registry.catalog import get_entry, list_slugs
from intergrax.integrations.registry.factory import resolve
from tests.qualification.pdc_x.p6_ecapfix._wrapper_support import (
    RecordingFactory,
    install_recording_owning_factory,
    wrapper_depth,
)

pytestmark = pytest.mark.unit

_SLUG = "ollama"
_REGISTER_MODULE = "intergrax.integrations.providers.embedding_provider.ollama.register"


def _register_owning_row_then_embedding_contribution() -> None:
    register_ollama_integration()
    register_ollama_embedding_provider_integration()


def test_embedding_enabled_defaults_to_false(cleared_catalog: None) -> None:
    _register_owning_row_then_embedding_contribution()

    resolved = resolve(IntegrationCategory.EMBEDDING_PROVIDER, _SLUG)

    assert isinstance(resolved, OllamaEmbeddingProviderIntegration)
    assert resolved.config.enabled is False


def test_embedding_enabled_true_is_passed_through(cleared_catalog: None) -> None:
    _register_owning_row_then_embedding_contribution()

    resolved = resolve(IntegrationCategory.EMBEDDING_PROVIDER, _SLUG, config={"enabled": True})

    assert isinstance(resolved, OllamaEmbeddingProviderIntegration)
    assert resolved.config.enabled is True


def test_embedding_enabled_false_is_honoured(cleared_catalog: None) -> None:
    _register_owning_row_then_embedding_contribution()

    resolved = resolve(IntegrationCategory.EMBEDDING_PROVIDER, _SLUG, config={"enabled": False})

    assert isinstance(resolved, OllamaEmbeddingProviderIntegration)
    assert resolved.config.enabled is False


def test_embedding_category_does_not_call_the_owning_factory(cleared_catalog: None) -> None:
    register_ollama_integration()
    owning_factory = get_entry(_SLUG).factory
    recorder = RecordingFactory(lambda: owning_factory(integration_category=IntegrationCategory.MODEL_SERVING_RUNTIME))
    install_recording_owning_factory(_SLUG, recorder)
    register_ollama_embedding_provider_integration()

    resolved = resolve(IntegrationCategory.EMBEDDING_PROVIDER, _SLUG)

    assert isinstance(resolved, OllamaEmbeddingProviderIntegration)
    assert recorder.calls == []


def test_non_embedding_keyword_arguments_reach_the_owning_factory_unchanged(cleared_catalog: None) -> None:
    register_ollama_integration()
    owning_factory = get_entry(_SLUG).factory
    recorder = RecordingFactory(lambda: owning_factory(integration_category=IntegrationCategory.MODEL_SERVING_RUNTIME))
    install_recording_owning_factory(_SLUG, recorder)
    register_ollama_embedding_provider_integration()

    resolved = resolve(IntegrationCategory.MODEL_SERVING_RUNTIME, _SLUG, config={"enabled": False, "marker": 7})

    assert isinstance(resolved, OllamaModelServingRuntimeIntegration)
    assert recorder.calls == [
        {"integration_category": IntegrationCategory.MODEL_SERVING_RUNTIME, "enabled": False, "marker": 7}
    ]


def test_repeated_registration_does_not_nest_the_wrapper(cleared_catalog: None) -> None:
    _register_owning_row_then_embedding_contribution()
    assert wrapper_depth(get_entry(_SLUG).factory, module_name=_REGISTER_MODULE) == 1

    with pytest.raises(ValueError, match="already registered"):
        register_ollama_embedding_provider_integration()
    assert wrapper_depth(get_entry(_SLUG).factory, module_name=_REGISTER_MODULE) == 1

    register_ollama_integration(override=True)
    register_ollama_embedding_provider_integration(override=True)

    assert list_slugs().count(_SLUG) == 1
    assert wrapper_depth(get_entry(_SLUG).factory, module_name=_REGISTER_MODULE) == 1
