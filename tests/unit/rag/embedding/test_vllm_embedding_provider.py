# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from intergrax.rag.embedding.providers.vllm_embedding_provider import VllmEmbeddingProvider

pytestmark = pytest.mark.unit


def test_vllm_embedding_provider_name() -> None:
    provider = VllmEmbeddingProvider(model_name="BAAI/bge-small-en-v1.5")
    assert provider.provider_name() == "vllm"


@patch("intergrax.rag.embedding.providers.vllm_embedding_provider.OpenAIEmbeddings")
def test_vllm_embedding_embed_batch(mock_embeddings_cls: MagicMock) -> None:
    mock_model = MagicMock()
    mock_model.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_model.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_embeddings_cls.return_value = mock_model

    with patch.dict(
        "os.environ",
        {
            "INTERGRAX_DEFAULT_VLLM_EMBED_BASE_URL": "http://127.0.0.1:8101/v1",
            "VLLM_API_KEY": "EMPTY",
        },
        clear=False,
    ):
        provider = VllmEmbeddingProvider(model_name="BAAI/bge-small-en-v1.5")
        vectors = provider.embed(["hello", "world"])

    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32
    mock_embeddings_cls.assert_called_once_with(
        model="BAAI/bge-small-en-v1.5",
        openai_api_base="http://127.0.0.1:8101/v1",
        openai_api_key="EMPTY",
        check_embedding_ctx_length=False,
    )


@patch("intergrax.rag.embedding.providers.vllm_embedding_provider.OpenAIEmbeddings")
def test_vllm_embedding_empty_batch(mock_embeddings_cls: MagicMock) -> None:
    mock_model = MagicMock()
    mock_model.embed_query.return_value = [0.0, 0.0]
    mock_embeddings_cls.return_value = mock_model

    provider = VllmEmbeddingProvider(model_name="BAAI/bge-small-en-v1.5")
    vectors = provider.embed([])

    assert vectors.shape == (0, 2)
    mock_model.embed_documents.assert_not_called()


def test_default_registry_includes_vllm() -> None:
    from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_registry

    registry = create_default_registry()
    provider = registry.get("vllm")
    assert provider.provider_name() == "vllm"
