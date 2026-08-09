from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from intergrax.rag.embedding.providers.openai_embedding_provider import (
    OpenAIEmbeddingProvider,
)


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def dummy_openai_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")


def make_client(*vectors: list[float]) -> MagicMock:
    client = MagicMock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=index, embedding=vector)
            for index, vector in enumerate(vectors)
        ]
    )
    return client


def test_openai_provider_name_and_model_precedence() -> None:
    with patch.dict(
        "os.environ",
        {"INTERGRAX_DEFAULT_OPENAI_EMBED_MODEL": "env-model"},
        clear=False,
    ):
        assert OpenAIEmbeddingProvider()._model_name == "env-model"
        assert (
            OpenAIEmbeddingProvider(model_name="constructor-model")._model_name
            == "constructor-model"
        )

    assert OpenAIEmbeddingProvider().provider_name() == "openai"
    assert OpenAIEmbeddingProvider.DEFAULT_MODEL == "text-embedding-3-small"


@patch("intergrax.rag.embedding.providers.openai_embedding_provider.OpenAI")
def test_openai_client_is_lazy_and_uses_sdk_credentials(
    mock_openai_cls: MagicMock,
) -> None:
    client = make_client([0.1, 0.2])
    mock_openai_cls.return_value = client
    provider = OpenAIEmbeddingProvider()

    mock_openai_cls.assert_not_called()
    provider.embed(["hello"])

    mock_openai_cls.assert_called_once_with()
    client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["hello"],
    )


@patch("intergrax.rag.embedding.providers.openai_embedding_provider.OpenAI")
def test_openai_dimension_probe_is_cached(mock_openai_cls: MagicMock) -> None:
    client = make_client([0.1, 0.2, 0.3])
    mock_openai_cls.return_value = client
    provider = OpenAIEmbeddingProvider(model_name="custom-model")

    assert provider.dimension() == 3
    assert provider.dimension() == 3

    client.embeddings.create.assert_called_once_with(
        model="custom-model",
        input=["probe-dimension"],
    )


@patch("intergrax.rag.embedding.providers.openai_embedding_provider.OpenAI")
def test_openai_embed_sets_dimension_and_returns_float32(
    mock_openai_cls: MagicMock,
) -> None:
    client = make_client([0.1, 0.2], [0.3, 0.4])
    mock_openai_cls.return_value = client
    provider = OpenAIEmbeddingProvider()

    vectors = provider.embed(["a", "b"])

    assert vectors.shape == (2, 2)
    assert vectors.dtype == np.float32
    assert provider.dimension() == 2
    assert client.embeddings.create.call_count == 1


@patch("intergrax.rag.embedding.providers.openai_embedding_provider.OpenAI")
def test_openai_empty_batch_resolves_dimension_without_empty_request(
    mock_openai_cls: MagicMock,
) -> None:
    client = make_client([0.0, 0.0, 0.0])
    mock_openai_cls.return_value = client
    provider = OpenAIEmbeddingProvider()

    vectors = provider.embed([])

    assert vectors.shape == (0, 3)
    assert vectors.dtype == np.float32
    client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["probe-dimension"],
    )
