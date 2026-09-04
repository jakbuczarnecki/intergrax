# © Artur Czarnecki. All rights reserved.

"""NeMo Guardrails llm_guardrail typed integration."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail._typed_integration import GuardrailTypedIntegration
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

NEMO_GUARDRAILS_PROVIDER_ID = "nemo_guardrails"


class NemoGuardrailsLlmGuardrailIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for NeMo Guardrails guardrail integration."""

    pass


class NemoGuardrailsLlmGuardrailIntegration(GuardrailTypedIntegration):
    """Provider-owned typed Integration for NeMo Guardrails."""

    config: NemoGuardrailsLlmGuardrailIntegrationConfig = NemoGuardrailsLlmGuardrailIntegrationConfig()
