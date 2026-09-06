# © Artur Czarnecki. All rights reserved.

"""P2-002-B3 — RAG embedding runtime cutover to IntegrationProfile."""

from __future__ import annotations

import ast
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
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
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.embedding.bootstrap.default_embedding_engine import (
    create_default_embedding_pipeline,
    create_default_registry,
)
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.contracts.runtime_binding import (
    EmbeddingProviderConfigurationError,
    EmbeddingProviderRuntimeBindingContext,
    EmbeddingProviderRuntimeBindingError,
    EmbeddingProviderRuntimeBinder,
)
from intergrax.rag.embedding.contracts.runtime_binding_spec import (
    EmbeddingProviderRuntimeBindingSpec,
)
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.pipeline.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig
from intergrax.rag.embedding.registry.profile import EmbeddingProfile, embedding_profile_from_env
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

_EXPECTED_PROVIDERS = frozenset(EMBEDDING_PROVIDER_SLUGS)


@pytest.fixture(autouse=True)
def _bootstrap_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    register_default_integrations(preset="full")
    yield
    clear_catalog()
    reset_default_integrations_state()


@dataclass
class _RecordedBindCall:
    provider_slug: str
    model: str | None
    execution_config: EmbeddingProviderExecutionConfig | None
    integration_options: dict[str, object] = field(default_factory=dict)


class _RecordingEmbeddingProvider(EmbeddingProvider):
    def __init__(self, provider_id: str) -> None:
        self._provider_id = provider_id

    def provider_name(self) -> str:
        return self._provider_id

    def dimension(self) -> int:
        return 3

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 3), dtype=np.float32)


class _RecordingBinder:
    def __init__(self, provider_id: str, recorded: list[_RecordedBindCall]) -> None:
        self._provider_id = provider_id
        self._recorded = recorded

    def bind(self, context: EmbeddingProviderRuntimeBindingContext) -> EmbeddingProvider:
        self._recorded.append(
            _RecordedBindCall(
                provider_slug=context.provider_slug,
                model=context.model,
                execution_config=context.execution_config,
                integration_options=dict(context.integration_options),
            )
        )
        return _RecordingEmbeddingProvider(self._provider_id)


@pytest.mark.parametrize("slug", sorted(_EXPECTED_PROVIDERS))
def test_first_party_contract_spec_supports_runtime_binding(slug: str) -> None:
    entry = get_entry(slug)
    spec = next(item for item in entry.contract_specs if item.category == "embedding_provider")
    assert spec.supports_runtime_binding is True
    runtime_binding = spec.runtime_binding
    assert isinstance(runtime_binding, EmbeddingProviderRuntimeBindingSpec)
    assert isinstance(runtime_binding.binder, EmbeddingProviderRuntimeBinder)


def test_validate_embedding_provider_slug_uses_catalog_not_hardcoded_set() -> None:
    validate_embedding_provider_slug("openai")
    with pytest.raises(ValueError, match="unknown embedding provider slug"):
        validate_embedding_provider_slug("not-a-real-provider")


def test_resolve_provider_slug_precedence_integration_profile_over_compat() -> None:
    slug = resolve_embedding_provider_slug(
        integration_profile=IntegrationProfile(embedding_provider="openai"),
        embedding_profile=EmbeddingProfile(provider="ollama"),
    )
    assert slug == "openai"


def test_resolve_provider_slug_provider_id_over_compat_profile() -> None:
    slug = resolve_embedding_provider_slug(
        provider_id="hf",
        embedding_profile=EmbeddingProfile(provider="ollama"),
    )
    assert slug == "hf"


def test_resolve_provider_slug_conflicts_fail_fast() -> None:
    with pytest.raises(EmbeddingProviderConfigurationError, match="conflicting"):
        resolve_embedding_provider_slug(
            integration_profile=IntegrationProfile(embedding_provider="openai"),
            provider_id="hf",
        )
    with pytest.raises(ValueError, match="unknown embedding provider slug"):
        bind_embedding_provider(provider_id="missing-provider")


def test_bind_provider_without_runtime_binder_fails_closed() -> None:
    slug = "fake_embedding_no_runtime_binder"

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
        display_name="Fake No Binder",
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
        env_prefix="INTERGRAX_FAKE_NO_BINDER",
        description="runtime binder missing probe",
    )
    register_from_manifest(manifest, _factory, contract_specs=(spec,))

    with pytest.raises(EmbeddingProviderRuntimeBindingError, match="no runtime binder"):
        bind_embedding_provider(integration_profile=IntegrationProfile(embedding_provider=slug))


def test_pluginability_e2e_without_vendor_sdk() -> None:
    slug = "fake_embedding_b3_pluginability"

    class _FakeRuntimeProvider(EmbeddingProvider):
        def provider_name(self) -> str:
            return slug

        def dimension(self) -> int:
            return 2

        def embed(self, texts: list[str]) -> np.ndarray:
            return np.asarray([[float(len(text)), 1.0] for text in texts], dtype=np.float32)

    class _PluginIntegration(EmbeddingProviderIntegrationContract):
        pass

    class _PluginBinder:
        def bind(self, context: EmbeddingProviderRuntimeBindingContext) -> EmbeddingProvider:
            assert context.provider_slug == slug
            return _FakeRuntimeProvider()

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
        display_name="Fake B3 Pluginability",
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
        env_prefix="INTERGRAX_FAKE_B3_PLUGIN",
        description="B3 pluginability probe",
    )
    register_from_manifest(manifest, _factory, contract_specs=(spec,))

    provider = bind_embedding_provider(
        integration_profile=IntegrationProfile(embedding_provider=slug),
        embedding_profile=EmbeddingProfile(provider=slug, model="probe-model"),
    )
    engine = EmbeddingEngine(provider=provider)
    pipeline = EmbeddingPipeline(engine=engine, provider_id=slug)

    vectors = pipeline.embed_texts(["ab", "abcd"])
    assert vectors.shape == (2, 2)
    np.testing.assert_array_equal(vectors, [[2.0, 1.0], [4.0, 1.0]])


@pytest.mark.parametrize("slug", sorted(_EXPECTED_PROVIDERS))
def test_first_party_runtime_binding_forwards_model(monkeypatch: pytest.MonkeyPatch, slug: str) -> None:
    recorded: list[_RecordedBindCall] = []
    entry = get_entry(slug)
    spec = next(item for item in entry.contract_specs if item.category == "embedding_provider")
    runtime_binding = spec.runtime_binding
    assert isinstance(runtime_binding, EmbeddingProviderRuntimeBindingSpec)
    binder = runtime_binding.binder
    assert binder is not None

    def wrapped_bind(context: EmbeddingProviderRuntimeBindingContext) -> EmbeddingProvider:
        recorded.append(
            _RecordedBindCall(
                provider_slug=context.provider_slug,
                model=context.model,
                execution_config=context.execution_config,
                integration_options=dict(context.integration_options),
            )
        )
        return _RecordingEmbeddingProvider(slug)

    monkeypatch.setattr(binder, "bind", wrapped_bind)

    provider = bind_embedding_provider(
        integration_profile=IntegrationProfile(embedding_provider=slug),
        embedding_profile=EmbeddingProfile(provider=slug, model="domain-model"),
        execution_config=EmbeddingProviderExecutionConfig(device="cpu", batch_size=16),
    )
    assert provider.provider_name() == slug
    assert recorded[0].model == "domain-model"
    if slug == "hf":
        assert recorded[0].execution_config == EmbeddingProviderExecutionConfig(
            device="cpu",
            batch_size=16,
        )


def test_hf_runtime_binding_maps_integration_options(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[_RecordedBindCall] = []

    class _StubHfProvider(EmbeddingProvider):
        def __init__(self, model_name: str | None, *, device: str | None, batch_size: int) -> None:
            self.model_name = model_name
            self.device = device
            self.batch_size = batch_size

        def provider_name(self) -> str:
            return "hf"

        def dimension(self) -> int:
            return 4

        def embed(self, texts: list[str]) -> np.ndarray:
            return np.zeros((len(texts), 4), dtype=np.float32)

    entry = get_entry("hf")
    spec = next(item for item in entry.contract_specs if item.category == "embedding_provider")
    runtime_binding = spec.runtime_binding
    assert isinstance(runtime_binding, EmbeddingProviderRuntimeBindingSpec)
    binder = runtime_binding.binder
    assert binder is not None

    def _fake_bind(context: EmbeddingProviderRuntimeBindingContext) -> EmbeddingProvider:
        recorded.append(
            _RecordedBindCall(
                provider_slug=context.provider_slug,
                model=context.model,
                execution_config=context.execution_config,
                integration_options=dict(context.integration_options),
            )
        )
        device = context.execution_config.device if context.execution_config else None
        batch_size = context.execution_config.batch_size if context.execution_config else 32
        if context.integration_options.get("device") is not None:
            device = str(context.integration_options["device"])
        if context.integration_options.get("batch_size") is not None:
            batch_size = int(context.integration_options["batch_size"])
        return _StubHfProvider(context.model, device=device, batch_size=batch_size or 32)

    monkeypatch.setattr(binder, "bind", _fake_bind)

    provider = bind_embedding_provider(
        integration_profile=IntegrationProfile(
            embedding_provider="hf",
            options={"hf": {"device": "cuda", "batch_size": 64}},
        ),
        embedding_profile=EmbeddingProfile(provider="hf", model="BAAI/bge-m3"),
        execution_config=EmbeddingProviderExecutionConfig(device="cpu", batch_size=16),
    )
    assert isinstance(provider, _StubHfProvider)
    assert provider.model_name == "BAAI/bge-m3"
    assert provider.device == "cuda"
    assert provider.batch_size == 64


def test_create_default_pipeline_uses_integration_profile_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[_RecordedBindCall] = []

    def _patch_binder(slug: str) -> None:
        entry = get_entry(slug)
        spec = next(item for item in entry.contract_specs if item.category == "embedding_provider")
        runtime_binding = spec.runtime_binding
        assert isinstance(runtime_binding, EmbeddingProviderRuntimeBindingSpec)
        binder = runtime_binding.binder
        assert binder is not None
        monkeypatch.setattr(
            binder,
            "bind",
            _RecordingBinder(slug, recorded).bind,
            raising=False,
        )

    for slug in _EXPECTED_PROVIDERS:
        _patch_binder(slug)

    monkeypatch.setenv("INTERGRAX_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("INTERGRAX_EMBEDDING_MODEL", "text-embedding-3-large")

    pipeline = create_default_embedding_pipeline()
    pipeline.embed_texts(["probe"])

    assert recorded[0].provider_slug == "openai"
    assert recorded[0].model == "text-embedding-3-large"


def test_catalog_bootstrap_does_not_import_vendor_sdks() -> None:
    blocked = ("sentence_transformers", "openai", "langchain_ollama")
    for name in blocked:
        sys.modules.pop(name, None)

    clear_catalog()
    reset_default_integrations_state()
    register_default_integrations(preset="full")

    for name in blocked:
        assert name not in sys.modules


def test_canonical_resolver_source_has_no_central_vendor_dispatch() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    resolver_path = repo_root / "intergrax" / "rag" / "embedding" / "runtime" / "resolver.py"
    source = resolver_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = ast.unparse(node.test)
            for slug in _EXPECTED_PROVIDERS:
                assert slug not in test


def test_runtime_binding_modules_allow_rag_contract_imports_only() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    root = repo_root / "intergrax" / "integrations" / "providers" / "embedding_provider"
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if path.name in {"runtime_binding.py", "contract_spec.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "intergrax.rag.embedding" in text:
            violations.append(str(path.relative_to(repo_root)))
    assert violations == []


def test_integrations_embedding_packages_only_import_rag_via_runtime_binding() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    root = repo_root / "intergrax" / "integrations" / "providers" / "embedding_provider"
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if path.name != "runtime_binding.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "intergrax.rag.embedding.providers" not in text:
            violations.append(str(path.relative_to(repo_root)))
    assert not violations


def test_openai_runtime_binding_lazy_imports_openai_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.embedding_provider.openai.runtime_binding import (
        OpenaiEmbeddingProviderRuntimeBinder,
    )

    calls: list[str] = []

    class _StubOpenAI:
        def __init__(self) -> None:
            calls.append("constructed")

    class _StubProvider(EmbeddingProvider):
        def provider_name(self) -> str:
            return "openai"

        def dimension(self) -> int:
            return 3

        def embed(self, texts: list[str]) -> np.ndarray:
            return np.ones((len(texts), 3), dtype=np.float32)

    def _fake_openai_provider(model_name: str | None = None) -> _StubProvider:
        calls.append(f"provider:{model_name}")
        return _StubProvider()

    stub_module = type(sys)("stub")
    stub_module.OpenAIEmbeddingProvider = _fake_openai_provider
    monkeypatch.setitem(
        sys.modules,
        "intergrax.rag.embedding.providers.openai_embedding_provider",
        stub_module,
    )

    binder = OpenaiEmbeddingProviderRuntimeBinder()
    provider = binder.bind(
        EmbeddingProviderRuntimeBindingContext(provider_slug="openai", model="text-embedding-3-small")
    )
    assert provider.provider_name() == "openai"
    assert calls == ["provider:text-embedding-3-small"]
