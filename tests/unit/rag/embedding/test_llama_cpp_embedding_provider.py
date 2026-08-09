# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from intergrax.rag.embedding.providers.llama_cpp_embedding_provider import LlamaCppEmbeddingProvider

pytestmark = pytest.mark.unit


def test_llama_cpp_embedding_provider_name_and_default_model() -> None:
    provider = LlamaCppEmbeddingProvider()
    assert provider.provider_name() == "llama_cpp"
    assert provider._model_name == LlamaCppEmbeddingProvider.DEFAULT_MODEL


@patch("intergrax.rag.embedding.providers.llama_cpp_embedding_provider.OpenAI")
def test_llama_cpp_embedding_uses_native_batch_transport(
    mock_openai_cls: MagicMock,
) -> None:
    mock_model = MagicMock()
    mock_model.embeddings.create.return_value.data = [
        MagicMock(index=0, embedding=[0.1, 0.2, 0.3]),
        MagicMock(index=1, embedding=[0.4, 0.5, 0.6]),
    ]
    mock_openai_cls.return_value = mock_model

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
    mock_openai_cls.assert_called_once_with(
        base_url="http://127.0.0.1:8103/v1",
        api_key="EMPTY",
    )
    mock_model.embeddings.create.assert_called_once_with(
        model="default",
        input=["hello", "world"],
    )


@patch("intergrax.rag.embedding.providers.llama_cpp_embedding_provider.OpenAI")
def test_llama_cpp_embedding_empty_batch_and_dimension_cache(
    mock_openai_cls: MagicMock,
) -> None:
    mock_model = MagicMock()
    mock_model.embeddings.create.return_value.data = [
        MagicMock(index=0, embedding=[0.0, 0.0])
    ]
    mock_openai_cls.return_value = mock_model

    provider = LlamaCppEmbeddingProvider(model_name="default")
    vectors = provider.embed([])

    assert vectors.shape == (0, 2)
    assert vectors.dtype == np.float32
    assert mock_model.embeddings.create.call_count == 1
    assert provider.dimension() == 2
    assert mock_model.embeddings.create.call_count == 1


def test_llama_cpp_base_url_precedence_and_normalization() -> None:
    provider = LlamaCppEmbeddingProvider()
    with patch.dict(
        "os.environ",
        {
            "INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_BASE_URL": " primary/ ",
            "INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL": "fallback/",
        },
        clear=False,
    ):
        assert provider._resolve_base_url() == "primary"

    with patch.dict(
        "os.environ",
        {
            "INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_BASE_URL": "",
            "INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL": " fallback/ ",
        },
        clear=False,
    ):
        assert provider._resolve_base_url() == "fallback"


@patch("intergrax.rag.embedding.providers.llama_cpp_embedding_provider.OpenAI")
def test_llama_cpp_api_key_and_model_env_resolution(
    mock_openai_cls: MagicMock,
) -> None:
    mock_openai_cls.return_value = MagicMock()
    with patch.dict(
        "os.environ",
        {
            "INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_MODEL": "env-model",
            "LLAMA_CPP_API_KEY": "explicit-key",
        },
        clear=False,
    ):
        provider = LlamaCppEmbeddingProvider()
        provider._ensure_client()

    assert provider._model_name == "env-model"
    mock_openai_cls.assert_called_once_with(
        base_url=LlamaCppEmbeddingProvider.DEFAULT_BASE_URL,
        api_key="explicit-key",
    )


def test_default_registry_includes_llama_cpp() -> None:
    from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_registry

    registry = create_default_registry()
    provider = registry.get("llama_cpp")
    assert provider.provider_name() == "llama_cpp"
