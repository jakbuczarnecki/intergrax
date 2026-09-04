# © Artur Czarnecki. All rights reserved.

"""Factory helpers for LLM Guard guardrail adapter."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.bundles.llm_guard import create_llm_guard_backend
from intergrax.integrations.contracts.llm_guardrail import LlmGuardrailBackend
from intergrax.integrations.providers.llm_guardrail.llm_guard.integration import (
    LLM_GUARD_PROVIDER_ID,
    LlmGuardLlmGuardrailIntegration,
    LlmGuardLlmGuardrailIntegrationConfig,
)

__all__ = [
    "create_llm_guard_llm_guardrail",
    "create_llm_guard_llm_guardrail_integration",
]


def create_llm_guard_llm_guardrail_integration(
    *,
    backend: LlmGuardrailBackend | None = None,
    enabled: bool = False,
) -> LlmGuardLlmGuardrailIntegration:
    """Build a contract-based LLM Guard guardrail integration."""
    if backend is not None:
        return LlmGuardLlmGuardrailIntegration.from_backend(
            backend,
            provider_id=LLM_GUARD_PROVIDER_ID,
            display_name="LLM Guard",
            enabled=enabled,
            config=LlmGuardLlmGuardrailIntegrationConfig(enabled=enabled),
        )
    return LlmGuardLlmGuardrailIntegration.for_provider(
        provider_id=LLM_GUARD_PROVIDER_ID,
        display_name="LLM Guard",
        config=LlmGuardLlmGuardrailIntegrationConfig(enabled=enabled),
    )


def create_llm_guard_llm_guardrail(**_kwargs: object) -> LlmGuardLlmGuardrailIntegration:
    """Catalog factory for ``llm_guard`` / ``llm_guardrail``."""
    backend = create_llm_guard_backend()
    return LlmGuardLlmGuardrailIntegration.from_backend(
        backend,
        provider_id=LLM_GUARD_PROVIDER_ID,
        display_name="LLM Guard",
        enabled=True,
        config=LlmGuardLlmGuardrailIntegrationConfig(enabled=True),
    )
