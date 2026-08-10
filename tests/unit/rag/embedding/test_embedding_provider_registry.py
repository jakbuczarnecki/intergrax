# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from concurrent.futures import ThreadPoolExecutor
import sys
import threading
from types import ModuleType

import pytest
import numpy as np

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.registry.embedding_provider_registry import (
    EmbeddingProviderDependencyError,
    EmbeddingProviderRegistrationError,
    EmbeddingProviderRegistry,
    lazy_import_provider_factory,
)


pytestmark = pytest.mark.unit


class FakeEmbeddingProvider(EmbeddingProvider):

    def provider_name(self) -> str:
        return "fake"

    def dimension(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 4), dtype=np.float32)


def test_register_and_get_provider() -> None:

    registry = EmbeddingProviderRegistry()

    provider = FakeEmbeddingProvider()

    registry.register(provider)

    retrieved = registry.get("fake")

    assert retrieved is provider


def test_duplicate_provider_registration() -> None:

    registry = EmbeddingProviderRegistry()

    provider = FakeEmbeddingProvider()

    registry.register(provider)

    with pytest.raises(ValueError):
        registry.register(provider)


def test_unknown_provider() -> None:

    registry = EmbeddingProviderRegistry()

    with pytest.raises(RuntimeError):
        registry.get("unknown")


def test_default_provider() -> None:

    registry = EmbeddingProviderRegistry()

    provider = FakeEmbeddingProvider()

    registry.register(provider)

    default_id = registry.default_provider()

    assert default_id == "fake"


def test_default_provider_empty_registry() -> None:

    registry = EmbeddingProviderRegistry()

    with pytest.raises(RuntimeError):
        registry.default_provider()


def test_lazy_factory_reports_missing_dependency_with_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(_: str) -> object:
        raise ModuleNotFoundError("sentence-transformers is missing", name="sentence_transformers")

    monkeypatch.setattr(
        "intergrax.rag.embedding.registry.embedding_provider_registry.import_module",
        missing_dependency,
    )
    factory = lazy_import_provider_factory(
        provider_id="hf",
        module_name="intergrax.rag.embedding.providers.hf_embedding_provider",
        class_name="HFEmbeddingProvider",
        dependency_name="sentence-transformers",
        extra_name="rag-local-embeddings",
    )
    registry = EmbeddingProviderRegistry()
    registry.register_factory("hf", factory)

    with pytest.raises(
        EmbeddingProviderDependencyError,
        match=r"Intergrax-ai\[rag-local-embeddings\]",
    ):
        registry.get("hf")


def test_lazy_factory_reports_missing_provider_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = type("Module", (), {})()
    monkeypatch.setattr(
        "intergrax.rag.embedding.registry.embedding_provider_registry.import_module",
        lambda _: module,
    )
    factory = lazy_import_provider_factory(
        provider_id="broken",
        module_name="broken_provider",
        class_name="Provider",
        dependency_name="broken-package",
    )

    with pytest.raises(EmbeddingProviderRegistrationError, match="does not define"):
        factory()


def test_lazy_provider_first_initialization_is_concurrent_safe() -> None:
    registry = EmbeddingProviderRegistry()
    calls = 0
    calls_lock = threading.Lock()

    def factory() -> EmbeddingProvider:
        nonlocal calls
        with calls_lock:
            calls += 1
        return FakeEmbeddingProvider()

    registry.register_factory("fake", factory)
    with ThreadPoolExecutor(max_workers=2) as executor:
        providers = list(executor.map(lambda _: registry.get("fake"), range(2)))

    assert calls == 1
    assert providers[0] is providers[1]


def test_hf_provider_constructs_and_embeds_with_mock_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str | None = None) -> None:
            self.model_name = model_name
            self.device = device

        def get_sentence_embedding_dimension(self) -> int:
            return 2

        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            return np.asarray([[float(len(text)), 1.0] for text in texts])

    fake_sentence_transformers = ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)
    sys.modules.pop("intergrax.rag.embedding.providers.hf_embedding_provider", None)

    factory = lazy_import_provider_factory(
        provider_id="hf",
        module_name="intergrax.rag.embedding.providers.hf_embedding_provider",
        class_name="HFEmbeddingProvider",
        dependency_name="sentence-transformers",
        extra_name="rag-local-embeddings",
    )
    provider = factory()

    assert provider.provider_name() == "hf"
    np.testing.assert_array_equal(
        provider.embed(["a", "abcd"]),
        np.asarray([[1.0, 1.0], [4.0, 1.0]], dtype=np.float32),
    )