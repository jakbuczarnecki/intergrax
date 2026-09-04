# © Artur Czarnecki. All rights reserved.

"""OpenGuardrails llm_guardrail typed integration."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail._typed_integration import GuardrailTypedIntegration
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

OPENGUARDRAILS_PROVIDER_ID = "openguardrails"


class OpenguardrailsLlmGuardrailIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for OpenGuardrails guardrail integration."""

    pass


class OpenguardrailsLlmGuardrailIntegration(GuardrailTypedIntegration):
    """Provider-owned typed Integration for OpenGuardrails."""

    config: OpenguardrailsLlmGuardrailIntegrationConfig = OpenguardrailsLlmGuardrailIntegrationConfig()
