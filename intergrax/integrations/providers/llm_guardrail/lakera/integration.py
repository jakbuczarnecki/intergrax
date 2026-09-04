# © Artur Czarnecki. All rights reserved.

"""Lakera llm_guardrail typed integration."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail._typed_integration import GuardrailTypedIntegration
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

LAKERA_PROVIDER_ID = "lakera"


class LakeraLlmGuardrailIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Lakera guardrail integration."""

    pass


class LakeraLlmGuardrailIntegration(GuardrailTypedIntegration):
    """Provider-owned typed Integration for Lakera."""

    config: LakeraLlmGuardrailIntegrationConfig = LakeraLlmGuardrailIntegrationConfig()
