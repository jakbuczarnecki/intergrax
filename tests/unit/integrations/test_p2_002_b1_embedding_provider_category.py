# © Artur Czarnecki. All rights reserved.

"""P2-002-B1 — embedding_provider Integrations category foundation."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.registry.catalog import clear_catalog, get_entry
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.integrations.registry.factory import resolve_from_profile
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.integrations.categories import (
    PROVIDER_CATEGORY_CONTRACT_REGISTRY,
    EmbeddingProviderIntegrationContract,
)
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.runtime.integrations.categories.embedding import (
    EMBEDDING_PROVIDER_INTEGRATION_CONTRACT_SCHEMA,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationConfig,
    PlatformIntegrationKind,
    PlatformIntegrationSecurityPosture,
)
from intergrax.runtime.integrations.registry_v2 import build_integration_registration

pytestmark = pytest.mark.unit

_FAKE_EMBEDDING_SLUG = "fake_embedding_test"


class _FakeEmbeddingIntegration(EmbeddingProviderIntegrationContract):
    """Test-only embedding provider — no vendor semantics."""


class _SensitiveEmbeddingConfig(PlatformIntegrationConfig):
    api_key: str | None = None


def _fake_embedding_factory(*, enabled: bool = False) -> _FakeEmbeddingIntegration:
    return _FakeEmbeddingIntegration.for_provider(
        provider_id=_FAKE_EMBEDDING_SLUG,
        display_name="Fake Embedding Test",
        config=CategoryIntegrationConfig(enabled=enabled),
    )


def _fake_embedding_contract_spec():
    return declare_integration_contract(
        category="embedding_provider",
        provider_id=_FAKE_EMBEDDING_SLUG,
        integration_class=_FakeEmbeddingIntegration,
        contract_class=EmbeddingProviderIntegrationContract,
        contract_factory=_fake_embedding_factory,
        display_name="Fake Embedding Test",
        config_class=CategoryIntegrationConfig,
        capabilities=(
            PlatformIntegrationCapability.CONNECT,
            PlatformIntegrationCapability.READ,
            PlatformIntegrationCapability.HEALTH_CHECK,
        ),
        security_posture=PlatformIntegrationSecurityPosture(),
        supports_runtime_binding=True,
        supports_health_check=True,
        metadata={"source": "p2_002_b1_test"},
    )


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    yield
    clear_catalog()


def test_embedding_provider_category_enum_and_profile_field_mapping() -> None:
    assert IntegrationCategory.EMBEDDING_PROVIDER.value == "embedding_provider"
    from intergrax.integrations.contracts.base import PROFILE_FIELD_BY_CATEGORY

    assert PROFILE_FIELD_BY_CATEGORY[IntegrationCategory.EMBEDDING_PROVIDER.value] == "embedding_provider"
    assert "embedding_provider" in IntegrationProfile._SLUG_FIELDS


def test_provider_category_contract_registry_includes_embedding_provider() -> None:
    assert PROVIDER_CATEGORY_CONTRACT_REGISTRY["embedding_provider"] is EmbeddingProviderIntegrationContract


def test_embedding_category_contract_shape() -> None:
    contract = EmbeddingProviderIntegrationContract.for_provider(provider_id="example_embedding")
    assert contract.schema_id == EMBEDDING_PROVIDER_INTEGRATION_CONTRACT_SCHEMA
    assert contract.integration_kind == PlatformIntegrationKind.EMBEDDING_PROVIDER.value
    assert PlatformIntegrationCapability.CONNECT in contract.capabilities
    assert contract.config.enabled is False


def test_embedding_category_contract_public_view_does_not_expose_secrets() -> None:
    config = _SensitiveEmbeddingConfig(enabled=True, api_key="super-secret-key")
    contract = EmbeddingProviderIntegrationContract.for_provider(
        provider_id="example_embedding",
        config=config,
    )
    public_view = contract.public_view()
    serialized = json.dumps(public_view)
    assert "api_key" not in public_view["config"]
    assert "super-secret-key" not in serialized


def test_fake_embedding_provider_explicit_registration() -> None:
    manifest = IntegrationManifest(
        slug=_FAKE_EMBEDDING_SLUG,
        categories=(IntegrationCategory.EMBEDDING_PROVIDER,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_FAKE_EMBEDDING_TEST",
        description="fake embedding provider for P2-002-B1",
    )
    register_from_manifest(manifest, lambda **_: {}, contract_specs=(_fake_embedding_contract_spec(),))

    entry = get_entry(_FAKE_EMBEDDING_SLUG)
    assert entry is not None
    assert IntegrationCategory.EMBEDDING_PROVIDER in entry.categories
    assert any(spec.category == "embedding_provider" for spec in entry.contract_specs)

    registration = build_integration_registration(_FAKE_EMBEDDING_SLUG)
    assert registration.category == "embedding_provider"
    assert registration.integration_class is _FakeEmbeddingIntegration
    assert registration.contract_class is EmbeddingProviderIntegrationContract


def test_fake_embedding_provider_without_explicit_specs_fails_closed() -> None:
    manifest = IntegrationManifest(
        slug=_FAKE_EMBEDDING_SLUG,
        categories=(IntegrationCategory.EMBEDDING_PROVIDER,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_FAKE_EMBEDDING_TEST",
        description="fake embedding provider for P2-002-B1",
    )
    with pytest.raises(ValueError, match="requires explicit contract_specs for typed categories"):
        register_from_manifest(manifest, lambda **_: {})


def test_integration_profile_embedding_provider_binding_resolution() -> None:
    manifest = IntegrationManifest(
        slug=_FAKE_EMBEDDING_SLUG,
        categories=(IntegrationCategory.EMBEDDING_PROVIDER,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_FAKE_EMBEDDING_TEST",
        description="fake embedding provider for P2-002-B1",
    )
    register_from_manifest(manifest, lambda **_: {"provider": "fake"}, contract_specs=(_fake_embedding_contract_spec(),))

    profile = IntegrationProfile(embedding_provider=_FAKE_EMBEDDING_SLUG)
    binding = profile.binding_for_field("embedding_provider")
    assert binding is not None
    assert binding.resolved_slug() == _FAKE_EMBEDDING_SLUG
    assert profile.slug_for_category(IntegrationCategory.EMBEDDING_PROVIDER) == _FAKE_EMBEDDING_SLUG

    resolved = resolve_from_profile(profile, IntegrationCategory.EMBEDDING_PROVIDER)
    assert resolved == {"provider": "fake"}


def test_integration_profile_embedding_provider_accepts_prebuilt_instance() -> None:
    sentinel = object()
    profile = IntegrationProfile(embedding_provider=sentinel)
    assert profile.instance_for_category(IntegrationCategory.EMBEDDING_PROVIDER) is sentinel
    assert profile.slug_for_category(IntegrationCategory.EMBEDDING_PROVIDER) is None


def test_integration_profile_rejects_unknown_embedding_slug() -> None:
    with pytest.raises(ValidationError):
        IntegrationProfile(embedding_provider="not_a_real_embedding_provider_xyz")


def test_integration_profile_rejects_wrong_slug_category() -> None:
    from intergrax.integrations.registry.catalog_manifests import SQLITE

    with pytest.raises(ValidationError):
        IntegrationProfile(embedding_provider=SQLITE)
