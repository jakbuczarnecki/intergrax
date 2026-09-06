# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register all first-party embedding provider catalog rows (P2-002-B2)."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.hf.register import (
    register_hf_embedding_provider_integration,
)
from intergrax.integrations.providers.embedding_provider.llama_cpp.register import (
    register_llama_cpp_embedding_provider_integration,
)
from intergrax.integrations.providers.embedding_provider.ollama.register import (
    register_ollama_embedding_provider_integration,
)
from intergrax.integrations.providers.embedding_provider.openai.register import (
    register_openai_embedding_provider_integration,
)
from intergrax.integrations.providers.embedding_provider.vllm.register import (
    register_vllm_embedding_provider_integration,
)

EMBEDDING_PROVIDER_SLUGS: tuple[str, ...] = (
    "hf",
    "openai",
    "ollama",
    "vllm",
    "llama_cpp",
)


def register_embedding_provider_integrations(*, override: bool = False) -> None:
    """Register canonical embedding_provider catalog rows for all first-party slugs."""
    register_hf_embedding_provider_integration(override=override)
    register_vllm_embedding_provider_integration(override=override)
    register_llama_cpp_embedding_provider_integration(override=override)
    register_openai_embedding_provider_integration(override=override)
    register_ollama_embedding_provider_integration(override=override)
