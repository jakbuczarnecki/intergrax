# © Artur Czarnecki. All rights reserved.

"""P2-002-B2 — first-party embedding provider catalog registration."""

from __future__ import annotations

import inspect
import json
import sys
from importlib import import_module

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.embedding_provider.register_all import (
    EMBEDDING_PROVIDER_SLUGS,
    register_embedding_provider_integrations,
)
from intergrax.integrations.providers.layout import (
    SECONDARY_PROVIDER_CATEGORIES,
    SLUG_CATEGORY,
    categories_for_provider,
    provider_import_path,
)
from intergrax.integrations.registry.bootstrap import (
    register_default_integrations,
    reset_default_integrations_state,
)
from intergrax.integrations.registry.catalog import clear_catalog, get_entry, list_slugs
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.integrations.categories.embedding import EmbeddingProviderIntegrationContract
from intergrax.runtime.integrations.registry_v2 import (
    build_contract_registry_snapshot,
    build_integration_registration,
)

pytestmark = pytest.mark.unit

_EXPECTED_PROVIDERS: frozenset[str] = frozenset(
    {"hf", "openai", "ollama", "vllm", "llama_cpp"},
)


@pytest.fixture(autouse=True)
def _bootstrap_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    register_default_integrations(preset="full")
    yield
    clear_catalog()
    reset_default_integrations_state()


def test_expected_first_party_embedding_provider_set() -> None:
    assert frozenset(EMBEDDING_PROVIDER_SLUGS) == _EXPECTED_PROVIDERS
    registered = frozenset(list_slugs(category=IntegrationCategory.EMBEDDING_PROVIDER))
    assert _EXPECTED_PROVIDERS <= registered


@pytest.mark.parametrize("slug", sorted(_EXPECTED_PROVIDERS))
def test_embedding_provider_catalog_entry_has_explicit_contract_spec(slug: str) -> None:
    entry = get_entry(slug)
    assert IntegrationCategory.EMBEDDING_PROVIDER in entry.categories
    embedding_specs = [
        spec for spec in entry.contract_specs if spec.category == "embedding_provider"
    ]
    assert len(embedding_specs) == 1
    spec = embedding_specs[0]
    assert spec.provider_id == slug
    assert spec.contract_class is EmbeddingProviderIntegrationContract
    assert spec.supports_runtime_binding is True


@pytest.mark.parametrize("slug", sorted(_EXPECTED_PROVIDERS))
def test_embedding_provider_registry_v2_projection(slug: str) -> None:
    registration = build_integration_registration(slug, category="embedding_provider")
    assert registration.provider_id == slug
    assert registration.slug == slug
    assert registration.category == "embedding_provider"
    assert registration.supports_runtime_binding is True


def test_embedding_provider_registry_v2_snapshot_has_five_rows() -> None:
    registry = build_contract_registry_snapshot()
    rows = registry.list_by_category("embedding_provider")
    assert {row.provider_id for row in rows} == _EXPECTED_PROVIDERS


def test_layout_taxonomy_for_embedding_providers() -> None:
    assert SLUG_CATEGORY["hf"] == "embedding_provider"
    assert SLUG_CATEGORY["vllm"] == "embedding_provider"
    assert SLUG_CATEGORY["llama_cpp"] == "embedding_provider"
    assert categories_for_provider("openai") == ("managed_retrieval", "embedding_provider")
    assert categories_for_provider("ollama") == ("model_serving_runtime", "embedding_provider")
    assert SECONDARY_PROVIDER_CATEGORIES["openai"] == ("embedding_provider",)
    assert SECONDARY_PROVIDER_CATEGORIES["ollama"] == ("embedding_provider",)
    assert provider_import_path("hf") == "intergrax.integrations.providers.embedding_provider.hf"
    assert provider_import_path("openai", "embedding_provider") == (
        "intergrax.integrations.providers.embedding_provider.openai"
    )


@pytest.mark.parametrize("slug", sorted(_EXPECTED_PROVIDERS))
def test_provider_manifest_slug_matches_runtime_provider_id(slug: str) -> None:
    if slug in {"openai", "ollama"}:
        module = import_module(f"intergrax.integrations.providers.embedding_provider.{slug}.integration")
    else:
        module = import_module(f"intergrax.integrations.providers.embedding_provider.{slug}.integration")
    provider_id = None
    for name in dir(module):
        if name.endswith("_EMBEDDING_PROVIDER_ID"):
            provider_id = getattr(module, name)
            break
    assert provider_id == slug


def test_integration_profile_accepts_registered_embedding_slugs() -> None:
    for slug in sorted(_EXPECTED_PROVIDERS):
        profile = IntegrationProfile(embedding_provider=slug)
        assert profile.slug_for_category(IntegrationCategory.EMBEDDING_PROVIDER) == slug


def test_embedding_provider_package_has_explicit_public_exports() -> None:
    blocked = (
        "sentence_transformers",
        "openai",
        "langchain_ollama",
    )
    for name in blocked:
        sys.modules.pop(name, None)

    import intergrax.integrations.providers.embedding_provider as embedding_provider_package

    source = inspect.getsource(embedding_provider_package)
    assert "def __getattr__" not in source
    assert embedding_provider_package.EMBEDDING_PROVIDER_SLUGS == EMBEDDING_PROVIDER_SLUGS
    assert (
        embedding_provider_package.register_embedding_provider_integrations
        is register_embedding_provider_integrations
    )

    for name in blocked:
        assert name not in sys.modules


def test_catalog_registration_does_not_import_optional_vendor_sdks() -> None:
    blocked = (
        "sentence_transformers",
        "openai",
        "langchain_ollama",
    )
    for name in blocked:
        sys.modules.pop(name, None)

    clear_catalog()
    from intergrax.integrations.providers.embedding_provider.hf.register import (
        register_hf_embedding_provider_integration,
    )
    from intergrax.integrations.providers.embedding_provider.llama_cpp.register import (
        register_llama_cpp_embedding_provider_integration,
    )
    from intergrax.integrations.providers.embedding_provider.vllm.register import (
        register_vllm_embedding_provider_integration,
    )

    register_hf_embedding_provider_integration()
    register_vllm_embedding_provider_integration()
    register_llama_cpp_embedding_provider_integration()

    for name in blocked:
        assert name not in sys.modules


def test_no_secrets_in_embedding_provider_contract_spec_metadata() -> None:
    entry = get_entry("openai")
    spec = next(s for s in entry.contract_specs if s.category == "embedding_provider")
    serialized = json.dumps(dict(spec.metadata))
    assert "api_key" not in serialized
    assert "sk-" not in serialized


def test_sixth_provider_pluginability_without_core_edits() -> None:
    from intergrax.integrations.contracts.base import IntegrationStatus
    from intergrax.integrations.core.manifest import IntegrationManifest
    from intergrax.integrations.registry.contract_spec import declare_integration_contract
    from intergrax.integrations.registry.plugin_register import register_from_manifest
    from intergrax.runtime.integrations.contracts import (
        PlatformIntegrationCapability,
        PlatformIntegrationConfig,
        PlatformIntegrationSecurityPosture,
    )

    slug = "fake_embedding_pluginability_test"

    class _PluginIntegration(EmbeddingProviderIntegrationContract):
        pass

    def _factory(*, enabled: bool = False) -> _PluginIntegration:
        return _PluginIntegration.for_provider(
            provider_id=slug,
            config=PlatformIntegrationConfig(enabled=enabled),
        )

    spec = declare_integration_contract(
        category="embedding_provider",
        provider_id=slug,
        integration_class=_PluginIntegration,
        contract_class=EmbeddingProviderIntegrationContract,
        contract_factory=_factory,
        display_name="Fake Embedding Pluginability",
        config_class=PlatformIntegrationConfig,
        capabilities=(
            PlatformIntegrationCapability.CONNECT,
            PlatformIntegrationCapability.READ,
            PlatformIntegrationCapability.HEALTH_CHECK,
        ),
        security_posture=PlatformIntegrationSecurityPosture(),
        supports_runtime_binding=False,
        supports_health_check=False,
        metadata={"source": "p2_002_b2_pluginability_test"},
    )
    manifest = IntegrationManifest(
        slug=slug,
        categories=(IntegrationCategory.EMBEDDING_PROVIDER,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_FAKE_EMBEDDING_PLUGINABILITY",
        description="sixth provider structural pluginability probe",
    )
    register_from_manifest(manifest, _factory, contract_specs=(spec,))

    entry = get_entry(slug)
    assert any(s.category == "embedding_provider" for s in entry.contract_specs)
    registration = build_integration_registration(slug, category="embedding_provider")
    assert registration.provider_id == slug


def test_integrations_do_not_import_rag_embedding_packages() -> None:
    repo_root = pytest.importorskip("pathlib").Path(__file__).resolve().parents[3]
    root = repo_root / "intergrax" / "integrations" / "providers" / "embedding_provider"
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if path.name in {"runtime_binding.py", "contract_spec.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "intergrax.rag.embedding" in text:
            violations.append(str(path.relative_to(repo_root)))
    assert violations == []
