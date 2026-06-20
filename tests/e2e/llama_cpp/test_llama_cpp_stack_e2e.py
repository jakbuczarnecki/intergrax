# © Artur Czarnecki. All rights reserved.
"""
Local-only E2E verification for llama.cpp Docker stack.

Not collected by GitHub CI (``no_ci`` + ``e2e`` + ``network``). Run after:

  cd infra/docker/llama-cpp && ./verify.sh
  # or: infra/docker/llama-cpp/verify.ps1
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import urllib.error
import urllib.request
from langchain_core.documents import Document

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.pipeline.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.providers.llama_cpp_embedding_provider import LlamaCppEmbeddingProvider
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry
from testing_support.builder import require_llama_cpp_embed_reachable, require_llama_cpp_reachable

pytestmark = [pytest.mark.e2e, pytest.mark.no_ci, pytest.mark.network]

_VERIFY_MODE = os.getenv("INTERGRAX_LLAMA_CPP_VERIFY", "").strip() == "1"


def _chat_base_url() -> str:
    return (
        os.getenv("INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL", "").strip().rstrip("/")
        or "http://127.0.0.1:8102/v1"
    )


def _embed_base_url() -> str:
    return (
        os.getenv("INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_BASE_URL", "").strip().rstrip("/")
        or "http://127.0.0.1:8103/v1"
    )


def test_llama_cpp_models_endpoint_lists_served_model() -> None:
    require_llama_cpp_reachable(base_url=_chat_base_url(), hard_fail=_VERIFY_MODE)
    models_url = f"{_chat_base_url()}/models"
    with urllib.request.urlopen(models_url, timeout=10.0) as response:
        body = response.read().decode("utf-8")
    assert "data" in body or "object" in body


def test_llama_cpp_chat_adapter_completion() -> None:
    require_llama_cpp_reachable(base_url=_chat_base_url(), hard_fail=_VERIFY_MODE)
    model = os.getenv("INTERGRAX_DEFAULT_LLAMA_CPP_MODEL", "").strip() or "default"
    adapter = LLMAdapterRegistry.create(LLMProvider.LLAMA_CPP, model=model)
    response = adapter.generate_messages(
        [ChatMessage(role="user", content="Reply with one short word: ok")],
        max_tokens=32,
        run_id="e2e-llama-cpp-chat",
    )
    assert isinstance(response.content, str)
    assert len(response.content.strip()) > 0
    assert response.provider == LLMProvider.LLAMA_CPP.value


def test_llama_cpp_profile_create_adapter() -> None:
    require_llama_cpp_reachable(base_url=_chat_base_url(), hard_fail=_VERIFY_MODE)
    model = os.getenv("INTERGRAX_DEFAULT_LLAMA_CPP_MODEL", "").strip() or "default"
    profile = LLMProfile(provider=LLMProvider.LLAMA_CPP, model=model)
    adapter = profile.create_adapter()
    response = adapter.generate_messages(
        [ChatMessage(role="user", content="Say hello in one word.")],
        max_tokens=16,
        run_id="e2e-llama-cpp-profile",
    )
    assert response.content.strip()


def test_llama_cpp_embedding_pipeline_documents() -> None:
    require_llama_cpp_embed_reachable(base_url=_embed_base_url(), hard_fail=_VERIFY_MODE)

    registry = EmbeddingProviderRegistry()
    provider = LlamaCppEmbeddingProvider()
    registry.register(provider)

    engine = EmbeddingEngine(registry)
    pipeline = EmbeddingPipeline(engine=engine, provider_id=provider.provider_name())
    manager = EmbeddingManager(pipeline=pipeline)

    docs = [
        Document(page_content="Embeddings enable semantic search."),
        Document(page_content="Vector similarity powers RAG systems."),
    ]

    result = manager.embed_documents(docs)

    assert len(result.embeddings) == 2
    vector_0 = result.embeddings[0]
    vector_1 = result.embeddings[1]

    assert isinstance(vector_0, np.ndarray)
    assert isinstance(vector_1, np.ndarray)
    assert vector_0.ndim == 1
    assert vector_1.ndim == 1
    assert vector_0.shape[0] > 0
    assert vector_0.shape[0] == vector_1.shape[0]
    assert vector_0.shape[0] == provider.dimension()


def test_llama_cpp_embed_models_endpoint() -> None:
    require_llama_cpp_embed_reachable(base_url=_embed_base_url(), hard_fail=_VERIFY_MODE)
    models_url = f"{_embed_base_url()}/models"
    try:
        with urllib.request.urlopen(models_url, timeout=10.0) as response:
            assert response.status == 200
    except urllib.error.HTTPError as exc:
        pytest.fail(f"llama.cpp embed /models returned HTTP {exc.code}")
