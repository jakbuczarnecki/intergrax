# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from intergrax.rag.embedding.providers.llama_cpp_embedding_provider import LlamaCppEmbeddingProvider

pytestmark = pytest.mark.unit


def test_llama_cpp_embedding_provider_name() -> None:
    provider = LlamaCppEmbeddingProvider(model_name="default")
    assert provider.provider_name() == "llama_cpp"


@patch("intergrax.rag.embedding.providers.llama_cpp_embedding_provider.OpenAIEmbeddings")
def test_llama_cpp_embedding_embed_batch(mock_embeddings_cls: MagicMock) -> None:
    mock_model = MagicMock()
    mock_model.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_model.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_embeddings_cls.return_value = mock_model

    with patch.dict(
        "os.environ",
        {
            "INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_BASE_URL": "http://127.0.0.1:8103/v1",
            "LLAMA_CPP_API_KEY": "EMPTY",
        },
        clear=False,
    ):
        provider = LlamaCppEmbeddingProvider(model_name="default")
        vectors = provider.embed(["hello", "world"])

    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32
    mock_embeddings_cls.assert_called_once_with(
        model="default",
        openai_api_base="http://127.0.0.1:8103/v1",
        openai_api_key="EMPTY",
        check_embedding_ctx_length=False,
    )


@patch("intergrax.rag.embedding.providers.llama_cpp_embedding_provider.OpenAIEmbeddings")
def test_llama_cpp_embedding_empty_batch(mock_embeddings_cls: MagicMock) -> None:
    mock_model = MagicMock()
    mock_model.embed_query.return_value = [0.0, 0.0]
    mock_embeddings_cls.return_value = mock_model

    provider = LlamaCppEmbeddingProvider(model_name="default")
    vectors = provider.embed([])

    assert vectors.shape == (0, 2)
    mock_model.embed_documents.assert_not_called()


def test_default_registry_includes_llama_cpp() -> None:
    from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_registry

    registry = create_default_registry()
    provider = registry.get("llama_cpp")
    assert provider.provider_name() == "llama_cpp"
