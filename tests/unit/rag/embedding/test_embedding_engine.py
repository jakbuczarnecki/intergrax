# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
import numpy as np

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry


pytestmark = pytest.mark.unit


class FakeEmbeddingProvider(EmbeddingProvider):

    def provider_name(self) -> str:
        return "fake"

    def dimension(self) -> int:
        return 3

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 3), dtype=np.float32)


def test_engine_embeds_texts() -> None:

    registry = EmbeddingProviderRegistry()
    provider = FakeEmbeddingProvider()

    registry.register(provider)

    engine = EmbeddingEngine(registry)

    result = engine.embed(
        provider_id="fake",
        texts=["a", "b", "c"],
    )

    assert result.shape == (3, 3)
    assert np.all(result == 1.0)


def test_engine_single_text() -> None:

    registry = EmbeddingProviderRegistry()
    provider = FakeEmbeddingProvider()

    registry.register(provider)

    engine = EmbeddingEngine(registry)

    result = engine.embed(
        provider_id="fake",
        texts=["hello"],
    )

    assert result.shape == (1, 3)


def test_engine_unknown_provider() -> None:

    registry = EmbeddingProviderRegistry()

    engine = EmbeddingEngine(registry)

    with pytest.raises(RuntimeError):
        engine.embed(
            provider_id="unknown",
            texts=["test"],
        )