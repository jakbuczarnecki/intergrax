# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
import numpy as np

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry


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