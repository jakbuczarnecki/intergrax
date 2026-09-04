# © Artur Czarnecki. All rights reserved.

"""Catalog manifest for ``bedrock_guardrails`` llm_guardrail integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="bedrock_guardrails",
    categories=(IntegrationCategory.LLM_GUARDRAIL,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_BEDROCK_GUARDRAILS",
    description="Bedrock Guardrails LLM guardrail harness adapter (M.12)",
)
