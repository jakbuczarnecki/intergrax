# © Artur Czarnecki. All rights reserved.

"""P2-002-B3 closure — generic runtime-binding extension and validation authority."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, fields

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.providers.embedding_provider.register_all import (
    EMBEDDING_PROVIDER_SLUGS,
)
from intergrax.integrations.registry.bootstrap import (
    register_default_integrations,
    reset_default_integrations_state,
)
from intergrax.integrations.registry.catalog import clear_catalog, get_entry
from intergrax.integrations.registry.contract_spec import (
    IntegrationContractSpec,
    declare_integration_contract,
)
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.runtime_binding import IntegrationRuntimeBindingSpec
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.contracts.runtime_binding import (
    EmbeddingProviderRuntimeBindingContext,
    EmbeddingProviderRuntimeBindingError,
    EmbeddingProviderRuntimeBinder,
)
from intergrax.rag.embedding.contracts.runtime_binding_spec import (
    EmbeddingProviderRuntimeBindingSpec,
)
from intergrax.rag.embedding.registry.profile import EmbeddingProfile
from intergrax.rag.embedding.registry.provider_authority import validate_embedding_provider_slug
from intergrax.rag.embedding.runtime.resolver import (
    bind_embedding_provider,
    resolve_embedding_provider_slug,
)
from intergrax.runtime.integrations.categories.embedding import EmbeddingProviderIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationConfig,
    PlatformIntegrationSecurityPosture,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _bootstrap_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    register_default_integrations(preset="full")
    yield
    clear_catalog()
    reset_default_integrations_state()


@dataclass(frozen=True)
class _OtherRuntimeBindingSpec(IntegrationRuntimeBindingSpec):
    marker: str = "other"


def _embedding_binder(spec: IntegrationContractSpec) -> EmbeddingProviderRuntimeBinder:
    runtime_binding = spec.runtime_binding
    assert isinstance(runtime_binding, EmbeddingProviderRuntimeBindingSpec)
    return runtime_binding.binder


def test_integration_contract_spec_carries_generic_runtime_binding_extension() -> None:
    binding = _OtherRuntimeBindingSpec()
    spec = declare_integration_contract(
        category="embedding_provider",
        provider_id="probe_runtime_binding",
        integration_class=EmbeddingProviderIntegrationContract,
        contract_class=EmbeddingProviderIntegrationContract,
        contract_factory=lambda **_: EmbeddingProviderIntegrationContract.for_provider(
            provider_id="probe_runtime_binding",
            config=PlatformIntegrationConfig(enabled=False),
        ),
        display_name="Probe",
        config_class=PlatformIntegrationConfig,
        capabilities=(PlatformIntegrationCapability.CONNECT,),
        security_posture=PlatformIntegrationSecurityPosture(),
        runtime_binding=binding,
    )
    assert spec.runtime_binding is binding
    assert isinstance(spec.runtime_binding, IntegrationRuntimeBindingSpec)


def test_generic_contract_spec_has_no_embedding_specific_field() -> None:
    field_names = {field.name for field in fields(IntegrationContractSpec)}
    assert "embedding_runtime_binder" not in field_names
    assert "runtime_binding" in field_names

    signature = inspect.signature(declare_integration_contract)
    assert "embedding_runtime_binder" not in signature.parameters
    assert "runtime_binding" in signature.parameters


def test_embedding_runtime_descriptor_contains_typed_binder() -> None:
    for slug in sorted(EMBEDDING_PROVIDER_SLUGS):
        entry = get_entry(slug)
        spec = next(item for item in entry.contract_specs if item.category == "embedding_provider")
        runtime_binding = spec.runtime_binding
        assert isinstance(runtime_binding, EmbeddingProviderRuntimeBindingSpec)
        assert isinstance(runtime_binding.binder, EmbeddingProviderRuntimeBinder)


def test_resolver_rejects_incompatible_runtime_binding_descriptor() -> None:
    slug = "fake_embedding_wrong_runtime_binding"

    class _Integration(EmbeddingProviderIntegrationContract):
        pass

    def _factory(*, enabled: bool = False) -> _Integration:
        return _Integration.for_provider(
            provider_id=slug,
            config=PlatformIntegrationConfig(enabled=enabled),
        )

    spec = declare_integration_contract(
        category="embedding_provider",
        provider_id=slug,
        integration_class=_Integration,
        contract_class=EmbeddingProviderIntegrationContract,
        contract_factory=_factory,
        display_name="Fake Wrong Binding",
        config_class=PlatformIntegrationConfig,
        capabilities=(
            PlatformIntegrationCapability.CONNECT,
            PlatformIntegrationCapability.READ,
        ),
        security_posture=PlatformIntegrationSecurityPosture(),
        supports_runtime_binding=True,
        supports_health_check=False,
        runtime_binding=_OtherRuntimeBindingSpec(),
    )
    manifest = IntegrationManifest(
        slug=slug,
        categories=(IntegrationCategory.EMBEDDING_PROVIDER,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_FAKE_WRONG_BINDING",
        description="runtime binding type mismatch probe",
    )
    register_from_manifest(manifest, _factory, contract_specs=(spec,))

    with pytest.raises(
        EmbeddingProviderRuntimeBindingError,
        match="not compatible with embedding_provider",
    ):
        bind_embedding_provider(integration_profile=IntegrationProfile(embedding_provider=slug))


def test_resolver_missing_runtime_binding_descriptor_fails_closed() -> None:
    slug = "fake_embedding_missing_runtime_binding"

    class _Integration(EmbeddingProviderIntegrationContract):
        pass

    def _factory(*, enabled: bool = False) -> _Integration:
        return _Integration.for_provider(
            provider_id=slug,
            config=PlatformIntegrationConfig(enabled=enabled),
        )

    spec = declare_integration_contract(
        category="embedding_provider",
        provider_id=slug,
        integration_class=_Integration,
        contract_class=EmbeddingProviderIntegrationContract,
        contract_factory=_factory,
        display_name="Fake Missing Binding",
        config_class=PlatformIntegrationConfig,
        capabilities=(
            PlatformIntegrationCapability.CONNECT,
            PlatformIntegrationCapability.READ,
        ),
        security_posture=PlatformIntegrationSecurityPosture(),
        supports_runtime_binding=True,
        supports_health_check=False,
        runtime_binding=None,
    )
    manifest = IntegrationManifest(
        slug=slug,
        categories=(IntegrationCategory.EMBEDDING_PROVIDER,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_FAKE_MISSING_BINDING",
        description="runtime binding missing probe",
    )
    register_from_manifest(manifest, _factory, contract_specs=(spec,))

    with pytest.raises(EmbeddingProviderRuntimeBindingError, match="no runtime binder"):
        bind_embedding_provider(integration_profile=IntegrationProfile(embedding_provider=slug))


def test_generic_contract_spec_module_has_no_rag_embedding_dependency() -> None:
    import intergrax.integrations.registry.contract_spec as contract_spec_module

    source = inspect.getsource(contract_spec_module)
    assert "rag.embedding" not in source
    assert "EmbeddingProviderRuntimeBinder" not in source
    assert "embedding_runtime_binder" not in source


def test_single_validate_embedding_provider_slug_implementation_in_runtime_scope() -> None:
    import intergrax.rag.embedding.registry.provider_authority as provider_authority
    import intergrax.rag.embedding.runtime.resolver as resolver_module

    assert inspect.getsource(provider_authority.validate_embedding_provider_slug)
    resolver_source = inspect.getsource(resolver_module)
    assert "def validate_embedding_provider_slug" not in resolver_source


def test_embedding_profile_and_resolver_use_catalog_backed_validation() -> None:
    slug = "fake_embedding_validation_plugin"

    class _Integration(EmbeddingProviderIntegrationContract):
        pass

    class _PluginBinder:
        def bind(self, context: EmbeddingProviderRuntimeBindingContext) -> EmbeddingProvider:
            class _Provider(EmbeddingProvider):
                def provider_name(self) -> str:
                    return slug

                def dimension(self) -> int:
                    return 2

                def embed(self, texts: list[str]) -> object:
                    return None

            return _Provider()

    def _factory(*, enabled: bool = False) -> _Integration:
        return _Integration.for_provider(
            provider_id=slug,
            config=PlatformIntegrationConfig(enabled=enabled),
        )

    spec = declare_integration_contract(
        category="embedding_provider",
        provider_id=slug,
        integration_class=_Integration,
        contract_class=EmbeddingProviderIntegrationContract,
        contract_factory=_factory,
        display_name="Fake Validation Plugin",
        config_class=PlatformIntegrationConfig,
        capabilities=(
            PlatformIntegrationCapability.CONNECT,
            PlatformIntegrationCapability.READ,
        ),
        security_posture=PlatformIntegrationSecurityPosture(),
        supports_runtime_binding=True,
        supports_health_check=False,
        runtime_binding=EmbeddingProviderRuntimeBindingSpec(binder=_PluginBinder()),
    )
    manifest = IntegrationManifest(
        slug=slug,
        categories=(IntegrationCategory.EMBEDDING_PROVIDER,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_FAKE_VALIDATION_PLUGIN",
        description="catalog validation plugin probe",
    )
    register_from_manifest(manifest, _factory, contract_specs=(spec,))

    assert validate_embedding_provider_slug(slug) == slug
    assert (
        resolve_embedding_provider_slug(
            integration_profile=IntegrationProfile(embedding_provider=slug),
        )
        == slug
    )
    profile = EmbeddingProfile(provider=slug, model="probe-model")
    assert profile.provider == slug

    with pytest.raises(ValueError, match="unknown embedding provider slug"):
        validate_embedding_provider_slug("not-a-real-provider")
