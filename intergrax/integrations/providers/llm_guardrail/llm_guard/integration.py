# © Artur Czarnecki. All rights reserved.

"""LLM Guard llm_guardrail typed integration."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail._typed_integration import GuardrailTypedIntegration
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

LLM_GUARD_PROVIDER_ID = "llm_guard"


class LlmGuardLlmGuardrailIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for LLM Guard guardrail integration."""

    pass


class LlmGuardLlmGuardrailIntegration(GuardrailTypedIntegration):
    """Provider-owned typed Integration for LLM Guard."""

    config: LlmGuardLlmGuardrailIntegrationConfig = LlmGuardLlmGuardrailIntegrationConfig()
