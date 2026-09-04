# © Artur Czarnecki. All rights reserved.

"""Presidio llm_guardrail typed integration."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail._typed_integration import GuardrailTypedIntegration
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PRESIDIO_PROVIDER_ID = "presidio"


class PresidioLlmGuardrailIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Presidio guardrail integration."""

    pass


class PresidioLlmGuardrailIntegration(GuardrailTypedIntegration):
    """Provider-owned typed Integration for Presidio."""

    config: PresidioLlmGuardrailIntegrationConfig = PresidioLlmGuardrailIntegrationConfig()
