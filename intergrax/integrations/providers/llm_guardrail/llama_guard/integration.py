# © Artur Czarnecki. All rights reserved.

"""Llama Guard llm_guardrail typed integration."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail._typed_integration import GuardrailTypedIntegration
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

LLAMA_GUARD_PROVIDER_ID = "llama_guard"


class LlamaGuardLlmGuardrailIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Llama Guard guardrail integration."""

    pass


class LlamaGuardLlmGuardrailIntegration(GuardrailTypedIntegration):
    """Provider-owned typed Integration for Llama Guard."""

    config: LlamaGuardLlmGuardrailIntegrationConfig = LlamaGuardLlmGuardrailIntegrationConfig()
