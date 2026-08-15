# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from intergrax.rag.embedding.bootstrap.default_embedding_engine import (
    create_default_embedding_pipeline,
    create_default_registry,
)
from intergrax.rag.embedding.providers.ollama_embedding_provider import OllamaEmbeddingProvider
from intergrax.rag.embedding.providers.openai_embedding_provider import OpenAIEmbeddingProvider
from intergrax.rag.embedding.registry.profile import EmbeddingProfile, embedding_profile_from_env


pytestmark = pytest.mark.unit


@contextmanager
def patch_openai_client():
    with patch("intergrax.rag.embedding.providers.openai_embedding_provider.OpenAI"):
        yield


def test_embedding_profile_from_env_defaults_to_ollama() -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv("INTERGRAX_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("INTERGRAX_EMBEDDING_MODEL", raising=False)
        profile = embedding_profile_from_env()
    assert profile.provider == "ollama"
    assert profile.model is None


def test_embedding_profile_from_env_reads_canonical_variables() -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("INTERGRAX_EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("INTERGRAX_EMBEDDING_MODEL", "text-embedding-3-large")
        profile = embedding_profile_from_env()
    assert profile == EmbeddingProfile(
        provider="openai",
        model="text-embedding-3-large",
    )


def test_embedding_profile_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown embedding provider"):
        EmbeddingProfile(provider="unknown_slug", model=None)


def test_create_default_pipeline_uses_embedding_provider_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("INTERGRAX_EMBEDDING_MODEL", "text-embedding-3-large")

    class SpyEngine:
        def embed(self, texts: list[str], *, provider_id: str) -> object:
            assert provider_id == "openai"
            return None

    pipeline = create_default_embedding_pipeline()
    pipeline._engine = SpyEngine()  # type: ignore[assignment]
    pipeline.embed_texts(["probe"])


def test_registry_factory_passes_embedding_model_to_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("INTERGRAX_EMBEDDING_MODEL", "profile-model")

    with patch_openai_client():
        registry = create_default_registry(embedding_model="profile-model")
        provider = registry.get("openai")
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider._model_name == "profile-model"


def test_ollama_provider_uses_constructor_model_not_legacy_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_DEFAULT_OLLAMA_EMBED_MODEL", "legacy-model")
    provider = OllamaEmbeddingProvider(model_name="canonical-model")
    assert provider._model_name == "canonical-model"


def test_openai_provider_uses_same_model_variable_via_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_DEFAULT_OPENAI_EMBED_MODEL", "legacy-model")
    with patch_openai_client():
        registry = create_default_registry(embedding_model="canonical-openai-model")
        provider = registry.get("openai")
        assert provider._model_name == "canonical-openai-model"


def test_lkw_env_example_contains_canonical_embedding_pair() -> None:
    from pathlib import Path

    text = Path(
        "applications/local_workspace_application/.env.example"
    ).read_text(encoding="utf-8")
    assert "INTERGRAX_EMBEDDING_PROVIDER=ollama" in text
    assert "INTERGRAX_EMBEDDING_MODEL=nomic-embed-text" in text
    assert "INTERGRAX_DEFAULT_OLLAMA_EMBED_MODEL" not in text


def test_compose_no_longer_exposes_legacy_ollama_embed_model_var() -> None:
    from pathlib import Path

    compose_dir = Path("applications/local_workspace_application/docker")
    for name in ("docker-compose.yml", "docker-compose.kafka.yml"):
        text = (compose_dir / name).read_text(encoding="utf-8")
        assert "INTERGRAX_EMBEDDING_PROVIDER" in text
        assert "INTERGRAX_EMBEDDING_MODEL" in text
        assert "INTERGRAX_DEFAULT_OLLAMA_EMBED_MODEL" not in text
        assert "INTERGRAX_RAG_EMBEDDING_PROVIDER" not in text


def test_root_env_example_documents_generation_and_embedding_sections() -> None:
    from pathlib import Path

    text = Path(".env.example").read_text(encoding="utf-8")
    assert "INTERGRAX_LLM_PROVIDER=ollama" in text
    assert "INTERGRAX_EMBEDDING_PROVIDER=ollama" in text
    assert "INTERGRAX_EMBEDDING_MODEL=nomic-embed-text" in text
    assert "INTERGRAX_DEFAULT_OLLAMA_EMBED_MODEL" not in text
    assert "INTERGRAX_DEFAULT_OPENAI_EMBED_MODEL" not in text
    assert "INTERGRAX_DEFAULT_HF_EMBED_MODEL" not in text
    assert "INTERGRAX_DEFAULT_VLLM_EMBED_MODEL" not in text
    assert "INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_MODEL" not in text


def test_llm_profile_remains_independent_from_embedding_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.llm_adapters.registry.profile import llm_profile_from_env

    monkeypatch.setenv("INTERGRAX_LLM_PROVIDER", "openai")
    monkeypatch.setenv("INTERGRAX_LLM_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("INTERGRAX_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("INTERGRAX_EMBEDDING_MODEL", "nomic-embed-text")

    llm = llm_profile_from_env()
    embedding = embedding_profile_from_env()

    assert llm.provider.value == "openai"
    assert llm.model == "gpt-4.1-mini"
    assert embedding.provider == "ollama"
    assert embedding.model == "nomic-embed-text"
