# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
import numpy as np

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine


pytestmark = pytest.mark.unit


class FakeEmbeddingProvider(EmbeddingProvider):

    def provider_name(self) -> str:
        return "fake"

    def dimension(self) -> int:
        return 3

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 3), dtype=np.float32)


def test_engine_embeds_texts() -> None:

    provider = FakeEmbeddingProvider()
    engine = EmbeddingEngine(provider=provider)

    result = engine.embed(
        texts=["a", "b", "c"],
    )

    assert result.shape == (3, 3)
    assert np.all(result == 1.0)


def test_engine_single_text() -> None:

    provider = FakeEmbeddingProvider()
    engine = EmbeddingEngine(provider=provider)

    result = engine.embed(
        texts=["hello"],
    )

    assert result.shape == (1, 3)


def test_engine_empty_input_returns_zero_row_matrix() -> None:

    provider = FakeEmbeddingProvider()
    engine = EmbeddingEngine(provider=provider)

    result = engine.embed(texts=[])

    assert result.shape == (0, 3)
