# © Artur Czarnecki. All rights reserved.

"""Factory helpers for Azure Content Safety guardrail adapter."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.bundles.http_guardrail import create_azure_content_safety_backend
from intergrax.integrations.contracts.llm_guardrail import LlmGuardrailBackend
from intergrax.integrations.providers.llm_guardrail.azure_content_safety.integration import (
    AZURE_CONTENT_SAFETY_PROVIDER_ID,
    AzureContentSafetyLlmGuardrailIntegration,
    AzureContentSafetyLlmGuardrailIntegrationConfig,
)

__all__ = [
    "create_azure_content_safety_llm_guardrail",
    "create_azure_content_safety_llm_guardrail_integration",
]


def create_azure_content_safety_llm_guardrail_integration(
    *,
    backend: LlmGuardrailBackend | None = None,
    enabled: bool = False,
) -> AzureContentSafetyLlmGuardrailIntegration:
    """Build a contract-based Azure Content Safety guardrail integration."""
    if backend is not None:
        return AzureContentSafetyLlmGuardrailIntegration.from_backend(
            backend,
            provider_id=AZURE_CONTENT_SAFETY_PROVIDER_ID,
            display_name="Azure Content Safety",
            enabled=enabled,
            config=AzureContentSafetyLlmGuardrailIntegrationConfig(enabled=enabled),
        )
    return AzureContentSafetyLlmGuardrailIntegration.for_provider(
        provider_id=AZURE_CONTENT_SAFETY_PROVIDER_ID,
        display_name="Azure Content Safety",
        config=AzureContentSafetyLlmGuardrailIntegrationConfig(enabled=enabled),
    )


def create_azure_content_safety_llm_guardrail(**_kwargs: object) -> AzureContentSafetyLlmGuardrailIntegration:
    """Catalog factory for ``azure_content_safety`` / ``llm_guardrail``."""
    backend = create_azure_content_safety_backend()
    return AzureContentSafetyLlmGuardrailIntegration.from_backend(
        backend,
        provider_id=AZURE_CONTENT_SAFETY_PROVIDER_ID,
        display_name="Azure Content Safety",
        enabled=True,
        config=AzureContentSafetyLlmGuardrailIntegrationConfig(enabled=True),
    )
