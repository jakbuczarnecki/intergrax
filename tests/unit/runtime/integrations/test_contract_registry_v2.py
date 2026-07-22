# © Artur Czarnecki. All rights reserved.

"""Contract registry v2 guards — INTEGRATIONS-3A."""

from __future__ import annotations

import sys
from collections.abc import Callable

import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.layout import SLUG_CATEGORY
from intergrax.runtime.integrations.categories import (
    OBSERVABILITY_BACKEND_CATEGORY,
    OBSERVABILITY_VENDOR_INTEGRATION_KIND,
    PROVIDER_CATEGORY_CONTRACT_REGISTRY,
)
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.contracts import PlatformIntegrationCapability, PlatformIntegrationContract
from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
)
from intergrax.runtime.integrations.registry_v2 import (
    DEFERRED_LLM_GUARDRAIL_SLUGS,
    DuplicateIntegrationRegistrationError,
    IntegrationRegistration,
    IntegrationRegistry,
    IntegrationRegistryError,
    build_contract_registry,
    build_integration_registration,
    non_deferred_provider_slugs,
)

pytestmark = pytest.mark.unit

_FORBIDDEN_VENDOR_SDK_MODULES = frozenset(
    {
        "boto3",
        "elasticsearch",
        "github",
        "hvac",
        "langfuse",
        "opentelemetry",
        "pinecone",
        "pymongo",
        "qdrant_client",
        "redis",
        "slack_sdk",
        "weaviate",
    }
)


class ElasticsearchVectorStoreIntegration(VectorStoreIntegrationContract):
    """Test-only provider/category registration for duplicate-key rules."""


class ElasticsearchObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """Test-only provider/category registration for duplicate-key rules."""


def _fake_vector_factory(*, enabled: bool = False) -> ElasticsearchVectorStoreIntegration:
    return ElasticsearchVectorStoreIntegration.for_provider(
        provider_id="elasticsearch",
        display_name="Elasticsearch Vector",
        config=CategoryIntegrationConfig(enabled=enabled),
    )


def _fake_observability_factory(*, enabled: bool = False) -> ElasticsearchObservabilityIntegration:
    return ElasticsearchObservabilityIntegration.for_provider(
        provider_id="elasticsearch",
        display_name="Elasticsearch Observability",
        config=ObservabilityVendorIntegrationConfig(enabled=enabled),
    )


def _registration_from_sample(
    *,
    slug: str,
    category: str,
    contract_class: type[PlatformIntegrationContract],
    integration_class: type[PlatformIntegrationContract],
    factory: Callable[..., PlatformIntegrationContract],
) -> IntegrationRegistration:
    sample = factory(enabled=False)
    return IntegrationRegistration(
        provider_id=sample.provider_id,
        slug=slug,
        category=category,
        integration_kind=sample.integration_kind,
        contract_class=contract_class,
        integration_class=integration_class,
        factory=factory,
        config_class=type(sample.config),
        display_name=sample.display_name or sample.provider_id,
        capabilities=tuple(capability.value for capability in sample.capabilities),
        security_posture=sample.security_posture,
        default_enabled=sample.enabled,
        supports_health_check=PlatformIntegrationCapability.HEALTH_CHECK.value
        in tuple(capability.value for capability in sample.capabilities),
    )


def test_registration_model_requires_identity_fields() -> None:
    with pytest.raises(IntegrationRegistryError, match="provider_id"):
        IntegrationRegistration(
            provider_id="",
            slug="qdrant",
            category="vector_store",
            integration_kind="vector_store",
            contract_class=VectorStoreIntegrationContract,
            integration_class=ElasticsearchVectorStoreIntegration,
            factory=_fake_vector_factory,
        )


def test_registry_rejects_duplicate_provider_category() -> None:
    registration = build_integration_registration("qdrant")
    registry = IntegrationRegistry([registration])

    with pytest.raises(DuplicateIntegrationRegistrationError):
        registry.register(registration)


def test_registry_allows_same_provider_across_categories() -> None:
    vector_registration = _registration_from_sample(
        slug="elasticsearch_vector",
        category="vector_store",
        contract_class=VectorStoreIntegrationContract,
        integration_class=ElasticsearchVectorStoreIntegration,
        factory=_fake_vector_factory,
    )
    observability_registration = _registration_from_sample(
        slug="elasticsearch_observability",
        category=OBSERVABILITY_BACKEND_CATEGORY,
        contract_class=ObservabilityVendorIntegrationContract,
        integration_class=ElasticsearchObservabilityIntegration,
        factory=_fake_observability_factory,
    )

    registry = IntegrationRegistry([vector_registration, observability_registration])

    assert registry.get(provider_id="elasticsearch", category="vector_store") is vector_registration
    assert registry.get(provider_id="elasticsearch", category=OBSERVABILITY_BACKEND_CATEGORY) is observability_registration


def test_registration_validates_integration_subclasses_category_contract() -> None:
    with pytest.raises(IntegrationRegistryError, match="must derive"):
        IntegrationRegistration(
            provider_id="broken",
            slug="broken",
            category="vector_store",
            integration_kind="vector_store",
            contract_class=VectorStoreIntegrationContract,
            integration_class=PlatformIntegrationContract,
            factory=_fake_vector_factory,
        )


def test_registration_factory_disabled_returns_integration() -> None:
    registration = build_integration_registration("qdrant")

    integration = registration.factory(enabled=False)

    assert isinstance(integration, registration.integration_class)
    assert isinstance(integration, registration.contract_class)
    assert integration.provider_id == "qdrant"
    assert integration.enabled is False


@pytest.mark.parametrize("slug", ["filesystem", "github", "langfuse", "qdrant", "slack"])
def test_registration_enabled_without_client_or_transport_raises(slug: str) -> None:
    registration = build_integration_registration(slug)

    with pytest.raises(IntegrationConfigurationError):
        registration.factory(enabled=True)


def test_observability_backend_registration_uses_observability_vendor_contract() -> None:
    registration = build_integration_registration("langfuse")

    assert registration.category == OBSERVABILITY_BACKEND_CATEGORY
    assert registration.integration_kind == OBSERVABILITY_VENDOR_INTEGRATION_KIND
    assert registration.contract_class is ObservabilityVendorIntegrationContract
    assert issubclass(registration.integration_class, ObservabilityVendorIntegrationContract)


def test_registry_lists_by_category() -> None:
    qdrant = build_integration_registration("qdrant")
    slack = build_integration_registration("slack")
    registry = IntegrationRegistry([qdrant, slack])

    assert registry.list_by_category("vector_store") == (qdrant,)
    assert registry.list_by_category("notification_channel") == (slack,)


def test_registry_lists_by_provider() -> None:
    vector_registration = _registration_from_sample(
        slug="elasticsearch_vector",
        category="vector_store",
        contract_class=VectorStoreIntegrationContract,
        integration_class=ElasticsearchVectorStoreIntegration,
        factory=_fake_vector_factory,
    )
    observability_registration = _registration_from_sample(
        slug="elasticsearch_observability",
        category=OBSERVABILITY_BACKEND_CATEGORY,
        contract_class=ObservabilityVendorIntegrationContract,
        integration_class=ElasticsearchObservabilityIntegration,
        factory=_fake_observability_factory,
    )
    registry = IntegrationRegistry([observability_registration, vector_registration])

    assert registry.list_by_provider("elasticsearch") == (
        observability_registration,
        vector_registration,
    )


def test_registry_construction_has_no_vendor_network_or_sdk_dependency() -> None:
    before_modules = set(sys.modules)

    build_contract_registry()

    new_modules = set(sys.modules) - before_modules
    forbidden_imports = sorted(
        module_name
        for module_name in new_modules
        if module_name in _FORBIDDEN_VENDOR_SDK_MODULES
        or any(module_name.startswith(f"{prefix}.") for prefix in _FORBIDDEN_VENDOR_SDK_MODULES)
    )
    assert forbidden_imports == []


def test_all_non_deferred_cutover_providers_are_registry_v2_compatible() -> None:
    registry = build_contract_registry()
    expected_slugs = set(non_deferred_provider_slugs())

    assert {registration.slug for registration in registry.list_all()} == expected_slugs
    for registration in registry.list_all():
        expected_contract = PROVIDER_CATEGORY_CONTRACT_REGISTRY[registration.category]
        assert registration.contract_class is expected_contract
        assert issubclass(registration.integration_class, expected_contract)
        assert registration.default_enabled is False
        if registration.category == "conversation_channel":
            assert registration.supports_runtime_binding is (registration.slug == "slack")
        else:
            assert registration.supports_runtime_binding is True

        integration = registration.factory(enabled=False)
        assert isinstance(integration, registration.integration_class)
        assert isinstance(integration, expected_contract)
        assert integration.enabled is False
        assert integration.provider_id == registration.provider_id
        assert integration.integration_kind == registration.integration_kind


def test_deferred_llm_guardrail_slugs_are_explicitly_excluded() -> None:
    registry = build_contract_registry()
    registered_slugs = {registration.slug for registration in registry.list_all()}

    for slug in DEFERRED_LLM_GUARDRAIL_SLUGS:
        assert slug not in registered_slugs
        assert SLUG_CATEGORY[slug] == "llm_guardrail"
