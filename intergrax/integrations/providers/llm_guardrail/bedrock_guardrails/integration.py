# © Artur Czarnecki. All rights reserved.

"""Bedrock Guardrails llm_guardrail typed integration."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail._typed_integration import GuardrailTypedIntegration
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

BEDROCK_GUARDRAILS_PROVIDER_ID = "bedrock_guardrails"


class BedrockGuardrailsLlmGuardrailIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Bedrock Guardrails guardrail integration."""

    pass


class BedrockGuardrailsLlmGuardrailIntegration(GuardrailTypedIntegration):
    """Provider-owned typed Integration for Bedrock Guardrails."""

    config: BedrockGuardrailsLlmGuardrailIntegrationConfig = BedrockGuardrailsLlmGuardrailIntegrationConfig()
