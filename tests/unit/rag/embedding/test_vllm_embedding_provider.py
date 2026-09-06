# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from intergrax.rag.embedding.providers.vllm_embedding_provider import VllmEmbeddingProvider

pytestmark = pytest.mark.unit


def test_vllm_embedding_provider_name_and_default_model(
) -> None:
    provider = VllmEmbeddingProvider()
    assert provider.provider_name() == "vllm"
    assert provider._model_name == VllmEmbeddingProvider.DEFAULT_MODEL


@patch("intergrax.rag.embedding.providers.vllm_embedding_provider.OpenAI")
def test_vllm_embedding_uses_native_batch_transport(
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
            "INTERGRAX_DEFAULT_VLLM_EMBED_BASE_URL": "http://127.0.0.1:8101/v1",
            "VLLM_API_KEY": "EMPTY",
        },
        clear=False,
    ):
        provider = VllmEmbeddingProvider(model_name="BAAI/bge-small-en-v1.5")
        vectors = provider.embed(["hello", "world"])

    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32
    mock_openai_cls.assert_called_once_with(
        base_url="http://127.0.0.1:8101/v1",
        api_key="EMPTY",
    )
    mock_model.embeddings.create.assert_called_once_with(
        model="BAAI/bge-small-en-v1.5",
        input=["hello", "world"],
    )


@patch("intergrax.rag.embedding.providers.vllm_embedding_provider.OpenAI")
def test_vllm_embedding_empty_batch_and_dimension_cache(
    mock_openai_cls: MagicMock,
) -> None:
    mock_model = MagicMock()
    mock_model.embeddings.create.return_value.data = [
        MagicMock(index=0, embedding=[0.0, 0.0])
    ]
    mock_openai_cls.return_value = mock_model

    provider = VllmEmbeddingProvider(model_name="BAAI/bge-small-en-v1.5")
    vectors = provider.embed([])

    assert vectors.shape == (0, 2)
    assert vectors.dtype == np.float32
    assert mock_model.embeddings.create.call_count == 1
    assert provider.dimension() == 2
    assert mock_model.embeddings.create.call_count == 1


def test_vllm_base_url_precedence_and_normalization() -> None:
    provider = VllmEmbeddingProvider()
    with patch.dict(
        "os.environ",
        {
            "INTERGRAX_DEFAULT_VLLM_EMBED_BASE_URL": " primary/ ",
            "INTERGRAX_DEFAULT_VLLM_BASE_URL": "fallback/",
        },
        clear=False,
    ):
        assert provider._resolve_base_url() == "primary"

    with patch.dict(
        "os.environ",
        {
            "INTERGRAX_DEFAULT_VLLM_EMBED_BASE_URL": "",
            "INTERGRAX_DEFAULT_VLLM_BASE_URL": " fallback/ ",
        },
        clear=False,
    ):
        assert provider._resolve_base_url() == "fallback"


@patch("intergrax.rag.embedding.providers.vllm_embedding_provider.OpenAI")
def test_vllm_api_key_resolution(mock_openai_cls: MagicMock) -> None:
    mock_openai_cls.return_value = MagicMock()
    with patch.dict(
        "os.environ",
        {
            "VLLM_API_KEY": "explicit-key",
        },
        clear=False,
    ):
        provider = VllmEmbeddingProvider(model_name="env-model")
        provider._ensure_client()

    assert provider._model_name == "env-model"
    mock_openai_cls.assert_called_once_with(
        base_url=VllmEmbeddingProvider.DEFAULT_BASE_URL,
        api_key="explicit-key",
    )


def test_vllm_provider_available_via_catalog_binding() -> None:
    from intergrax.integrations.registry.bootstrap import register_default_integrations
    from intergrax.integrations.registry.profile import IntegrationProfile
    from intergrax.rag.embedding.registry.profile import EmbeddingProfile
    from intergrax.rag.embedding.runtime.resolver import bind_embedding_provider

    register_default_integrations(preset="full")
    provider = bind_embedding_provider(
        integration_profile=IntegrationProfile(embedding_provider="vllm"),
        embedding_profile=EmbeddingProfile(provider="vllm", model="test-model"),
    )
    assert provider.provider_name() == "vllm"
