# © Artur Czarnecki. All rights reserved.

"""Factory helpers for Llama Guard guardrail adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from intergrax.integrations.providers.llm_guardrail.bundles.llama_guard import create_llama_guard_backend
from intergrax.integrations.contracts.llm_guardrail import LlmGuardrailBackend
from intergrax.integrations.providers.llm_guardrail.llama_guard.integration import (
    LLAMA_GUARD_PROVIDER_ID,
    LlamaGuardLlmGuardrailIntegration,
    LlamaGuardLlmGuardrailIntegrationConfig,
)

__all__ = [
    "create_llama_guard_llm_guardrail",
    "create_llama_guard_llm_guardrail_integration",
]


def create_llama_guard_llm_guardrail_integration(
    *,
    backend: LlmGuardrailBackend | None = None,
    enabled: bool = False,
) -> LlamaGuardLlmGuardrailIntegration:
    """Build a contract-based Llama Guard guardrail integration."""
    if backend is not None:
        return LlamaGuardLlmGuardrailIntegration.from_backend(
            backend,
            provider_id=LLAMA_GUARD_PROVIDER_ID,
            display_name="Llama Guard",
            enabled=enabled,
            config=LlamaGuardLlmGuardrailIntegrationConfig(enabled=enabled),
        )
    return LlamaGuardLlmGuardrailIntegration.for_provider(
        provider_id=LLAMA_GUARD_PROVIDER_ID,
        display_name="Llama Guard",
        config=LlamaGuardLlmGuardrailIntegrationConfig(enabled=enabled),
    )


def create_llama_guard_llm_guardrail(
    *,
    provider_options: Mapping[str, Any] | None = None,
    **kwargs: object,
) -> LlamaGuardLlmGuardrailIntegration:
    """Catalog factory for ``llama_guard`` / ``llm_guardrail``."""
    opts = provider_options
    if opts is None and kwargs:
        opts = dict(kwargs)
    backend = create_llama_guard_backend(provider_options=opts)
    return LlamaGuardLlmGuardrailIntegration.from_backend(
        backend,
        provider_id=LLAMA_GUARD_PROVIDER_ID,
        display_name="Llama Guard",
        enabled=True,
        config=LlamaGuardLlmGuardrailIntegrationConfig(enabled=True),
    )
