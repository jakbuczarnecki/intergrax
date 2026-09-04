# © Artur Czarnecki. All rights reserved.

"""Guardrails AI llm_guardrail typed integration."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail._typed_integration import GuardrailTypedIntegration
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GUARDRAILS_AI_PROVIDER_ID = "guardrails_ai"


class GuardrailsAiLlmGuardrailIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Guardrails AI guardrail integration."""

    pass


class GuardrailsAiLlmGuardrailIntegration(GuardrailTypedIntegration):
    """Provider-owned typed Integration for Guardrails AI."""

    config: GuardrailsAiLlmGuardrailIntegrationConfig = GuardrailsAiLlmGuardrailIntegrationConfig()
