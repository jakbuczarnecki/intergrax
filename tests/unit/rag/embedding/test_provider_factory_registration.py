# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Unit tests for provider-owned embedding factory registration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import pytest

from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_registry
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig
from intergrax.rag.embedding.registry.provider_factory_registration import (
    build_hf_provider_factory,
    build_llama_cpp_provider_factory,
    build_ollama_provider_factory,
    build_openai_provider_factory,
    build_vllm_provider_factory,
)

pytestmark = pytest.mark.unit


@dataclass
class _RecordedProviderCall:
    provider_id: str
    args: tuple[object, ...] = ()
    kwargs: dict[str, object] = field(default_factory=dict)


def _patch_recording_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider_id: str,
    recorded: list[_RecordedProviderCall],
) -> None:
    class RecordingProvider(EmbeddingProvider):
        def __init__(self, *args: object, **kwargs: object) -> None:
            recorded.append(
                _RecordedProviderCall(provider_id=provider_id, args=args, kwargs=kwargs)
            )

        def provider_name(self) -> str:
            return provider_id

        def dimension(self) -> int:
            return 4

        def embed(self, texts: list[str]) -> object:
            return None

    monkeypatch.setattr(
        "intergrax.rag.embedding.registry.provider_factory_registration.import_embedding_provider_class",
        lambda **_: cast(type[EmbeddingProvider], RecordingProvider),
    )


def test_hf_factory_maps_execution_config(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[_RecordedProviderCall] = []
    _patch_recording_provider(monkeypatch, provider_id="hf", recorded=recorded)

    factory = build_hf_provider_factory(
        embedding_model="BAAI/bge-m3",
        execution_config=EmbeddingProviderExecutionConfig(device="cuda", batch_size=64),
    )
    factory()

    assert recorded == [
        _RecordedProviderCall(
            provider_id="hf",
            args=("BAAI/bge-m3",),
            kwargs={"device": "cuda", "batch_size": 64},
        )
    ]


@pytest.mark.parametrize(
    ("builder", "provider_id"),
    [
        (build_openai_provider_factory, "openai"),
        (build_ollama_provider_factory, "ollama"),
        (build_vllm_provider_factory, "vllm"),
        (build_llama_cpp_provider_factory, "llama_cpp"),
    ],
)
def test_non_hf_factories_ignore_execution_config(
    monkeypatch: pytest.MonkeyPatch,
    builder: Callable[..., Callable[[], EmbeddingProvider]],
    provider_id: str,
) -> None:
    recorded: list[_RecordedProviderCall] = []
    _patch_recording_provider(monkeypatch, provider_id=provider_id, recorded=recorded)

    factory = builder(embedding_model="provider-model")
    factory()

    assert recorded == [
        _RecordedProviderCall(
            provider_id=provider_id,
            args=("provider-model",),
            kwargs={},
        )
    ]


def test_create_default_registry_hf_receives_execution_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[_RecordedProviderCall] = []
    _patch_recording_provider(monkeypatch, provider_id="hf", recorded=recorded)

    registry = create_default_registry(
        embedding_model="BAAI/bge-m3",
        execution_config=EmbeddingProviderExecutionConfig(device="cpu", batch_size=32),
    )
    registry.get("hf")

    assert recorded[0].kwargs == {"device": "cpu", "batch_size": 32}


@pytest.mark.parametrize("provider_id", ["openai", "ollama", "vllm", "llama_cpp"])
def test_create_default_registry_non_hf_receives_model_only(
    monkeypatch: pytest.MonkeyPatch,
    provider_id: str,
) -> None:
    recorded: list[_RecordedProviderCall] = []
    _patch_recording_provider(monkeypatch, provider_id=provider_id, recorded=recorded)

    registry = create_default_registry(
        embedding_model="provider-model",
        execution_config=EmbeddingProviderExecutionConfig(device="cuda", batch_size=64),
    )
    registry.get(provider_id)

    assert recorded[0].args == ("provider-model",)
    assert recorded[0].kwargs == {}
