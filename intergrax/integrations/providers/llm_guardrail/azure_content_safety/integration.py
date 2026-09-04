# © Artur Czarnecki. All rights reserved.

"""Azure Content Safety llm_guardrail typed integration."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail._typed_integration import GuardrailTypedIntegration
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AZURE_CONTENT_SAFETY_PROVIDER_ID = "azure_content_safety"


class AzureContentSafetyLlmGuardrailIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Azure Content Safety guardrail integration."""

    pass


class AzureContentSafetyLlmGuardrailIntegration(GuardrailTypedIntegration):
    """Provider-owned typed Integration for Azure Content Safety."""

    config: AzureContentSafetyLlmGuardrailIntegrationConfig = AzureContentSafetyLlmGuardrailIntegrationConfig()
