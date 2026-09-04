# © Artur Czarnecki. All rights reserved.

"""Factory helpers for Guardrails AI guardrail adapter."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.bundles.guardrails_ai import create_guardrails_ai_backend
from intergrax.integrations.contracts.llm_guardrail import LlmGuardrailBackend
from intergrax.integrations.providers.llm_guardrail.guardrails_ai.integration import (
    GUARDRAILS_AI_PROVIDER_ID,
    GuardrailsAiLlmGuardrailIntegration,
    GuardrailsAiLlmGuardrailIntegrationConfig,
)

__all__ = [
    "create_guardrails_ai_llm_guardrail",
    "create_guardrails_ai_llm_guardrail_integration",
]


def create_guardrails_ai_llm_guardrail_integration(
    *,
    backend: LlmGuardrailBackend | None = None,
    enabled: bool = False,
) -> GuardrailsAiLlmGuardrailIntegration:
    """Build a contract-based Guardrails AI guardrail integration."""
    if backend is not None:
        return GuardrailsAiLlmGuardrailIntegration.from_backend(
            backend,
            provider_id=GUARDRAILS_AI_PROVIDER_ID,
            display_name="Guardrails AI",
            enabled=enabled,
            config=GuardrailsAiLlmGuardrailIntegrationConfig(enabled=enabled),
        )
    return GuardrailsAiLlmGuardrailIntegration.for_provider(
        provider_id=GUARDRAILS_AI_PROVIDER_ID,
        display_name="Guardrails AI",
        config=GuardrailsAiLlmGuardrailIntegrationConfig(enabled=enabled),
    )


def create_guardrails_ai_llm_guardrail(**_kwargs: object) -> GuardrailsAiLlmGuardrailIntegration:
    """Catalog factory for ``guardrails_ai`` / ``llm_guardrail``."""
    backend = create_guardrails_ai_backend()
    return GuardrailsAiLlmGuardrailIntegration.from_backend(
        backend,
        provider_id=GUARDRAILS_AI_PROVIDER_ID,
        display_name="Guardrails AI",
        enabled=True,
        config=GuardrailsAiLlmGuardrailIntegrationConfig(enabled=True),
    )
