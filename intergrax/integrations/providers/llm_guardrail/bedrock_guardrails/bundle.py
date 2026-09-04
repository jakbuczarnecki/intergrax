# © Artur Czarnecki. All rights reserved.

"""Factory helpers for Bedrock Guardrails guardrail adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from intergrax.integrations.providers.llm_guardrail.bundles.bedrock_guardrails import create_bedrock_guardrails_backend
from intergrax.integrations.contracts.llm_guardrail import LlmGuardrailBackend
from intergrax.integrations.providers.llm_guardrail.bedrock_guardrails.integration import (
    BEDROCK_GUARDRAILS_PROVIDER_ID,
    BedrockGuardrailsLlmGuardrailIntegration,
    BedrockGuardrailsLlmGuardrailIntegrationConfig,
)

__all__ = [
    "create_bedrock_guardrails_llm_guardrail",
    "create_bedrock_guardrails_llm_guardrail_integration",
]


def create_bedrock_guardrails_llm_guardrail_integration(
    *,
    backend: LlmGuardrailBackend | None = None,
    enabled: bool = False,
) -> BedrockGuardrailsLlmGuardrailIntegration:
    """Build a contract-based Bedrock Guardrails guardrail integration."""
    if backend is not None:
        return BedrockGuardrailsLlmGuardrailIntegration.from_backend(
            backend,
            provider_id=BEDROCK_GUARDRAILS_PROVIDER_ID,
            display_name="Bedrock Guardrails",
            enabled=enabled,
            config=BedrockGuardrailsLlmGuardrailIntegrationConfig(enabled=enabled),
        )
    return BedrockGuardrailsLlmGuardrailIntegration.for_provider(
        provider_id=BEDROCK_GUARDRAILS_PROVIDER_ID,
        display_name="Bedrock Guardrails",
        config=BedrockGuardrailsLlmGuardrailIntegrationConfig(enabled=enabled),
    )


def create_bedrock_guardrails_llm_guardrail(
    *,
    provider_options: Mapping[str, Any] | None = None,
    **kwargs: object,
) -> BedrockGuardrailsLlmGuardrailIntegration:
    """Catalog factory for ``bedrock_guardrails`` / ``llm_guardrail``."""
    opts = provider_options
    if opts is None and kwargs:
        opts = dict(kwargs)
    backend = create_bedrock_guardrails_backend(provider_options=opts)
    return BedrockGuardrailsLlmGuardrailIntegration.from_backend(
        backend,
        provider_id=BEDROCK_GUARDRAILS_PROVIDER_ID,
        display_name="Bedrock Guardrails",
        enabled=True,
        config=BedrockGuardrailsLlmGuardrailIntegrationConfig(enabled=True),
    )
