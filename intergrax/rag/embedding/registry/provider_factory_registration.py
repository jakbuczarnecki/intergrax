# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Provider-owned factory registration for embedding providers.

TRANSITIONAL_RUNTIME_COMPATIBILITY — legacy RAG runtime factory map retained for B4 removal.
Canonical B3 runtime binding is owned by Integrations provider packages via
``runtime_binding`` on ``IntegrationContractSpec``.
"""

from __future__ import annotations

from collections.abc import Callable

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.registry.embedding_provider_registry import (
    import_embedding_provider_class,
)
from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig

_HF_MODULE = "intergrax.rag.embedding.providers.hf_embedding_provider"
_OPENAI_MODULE = "intergrax.rag.embedding.providers.openai_embedding_provider"
_OLLAMA_MODULE = "intergrax.rag.embedding.providers.ollama_embedding_provider"
_VLLM_MODULE = "intergrax.rag.embedding.providers.vllm_embedding_provider"
_LLAMA_CPP_MODULE = "intergrax.rag.embedding.providers.llama_cpp_embedding_provider"


def build_hf_provider_factory(
    *,
    embedding_model: str | None,
    execution_config: EmbeddingProviderExecutionConfig | None,
) -> Callable[[], EmbeddingProvider]:
    device = None if execution_config is None else execution_config.device
    batch_size = None if execution_config is None else execution_config.batch_size

    def create_provider() -> EmbeddingProvider:
        provider_type = import_embedding_provider_class(
            provider_id="hf",
            module_name=_HF_MODULE,
            class_name="HFEmbeddingProvider",
            dependency_name="sentence-transformers",
            extra_name="rag-local-embeddings",
        )
        if batch_size is not None:
            return provider_type(embedding_model, device=device, batch_size=batch_size)
        return provider_type(embedding_model, device=device)

    return create_provider


def build_openai_provider_factory(
    *,
    embedding_model: str | None,
) -> Callable[[], EmbeddingProvider]:
    return _build_model_name_only_factory(
        provider_id="openai",
        module_name=_OPENAI_MODULE,
        class_name="OpenAIEmbeddingProvider",
        dependency_name="openai",
        embedding_model=embedding_model,
    )


def build_ollama_provider_factory(
    *,
    embedding_model: str | None,
) -> Callable[[], EmbeddingProvider]:
    return _build_model_name_only_factory(
        provider_id="ollama",
        module_name=_OLLAMA_MODULE,
        class_name="OllamaEmbeddingProvider",
        dependency_name="langchain-ollama",
        embedding_model=embedding_model,
    )


def build_vllm_provider_factory(
    *,
    embedding_model: str | None,
) -> Callable[[], EmbeddingProvider]:
    return _build_model_name_only_factory(
        provider_id="vllm",
        module_name=_VLLM_MODULE,
        class_name="VllmEmbeddingProvider",
        dependency_name="openai",
        embedding_model=embedding_model,
    )


def build_llama_cpp_provider_factory(
    *,
    embedding_model: str | None,
) -> Callable[[], EmbeddingProvider]:
    return _build_model_name_only_factory(
        provider_id="llama_cpp",
        module_name=_LLAMA_CPP_MODULE,
        class_name="LlamaCppEmbeddingProvider",
        dependency_name="openai",
        embedding_model=embedding_model,
    )


def _build_model_name_only_factory(
    *,
    provider_id: str,
    module_name: str,
    class_name: str,
    dependency_name: str,
    embedding_model: str | None,
    extra_name: str | None = None,
) -> Callable[[], EmbeddingProvider]:
    def create_provider() -> EmbeddingProvider:
        provider_type = import_embedding_provider_class(
            provider_id=provider_id,
            module_name=module_name,
            class_name=class_name,
            dependency_name=dependency_name,
            extra_name=extra_name,
        )
        if embedding_model is not None:
            return provider_type(embedding_model)
        return provider_type()

    return create_provider
