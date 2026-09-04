# © Artur Czarnecki. All rights reserved.

"""Catalog manifest for ``guardrails_ai`` llm_guardrail integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="guardrails_ai",
    categories=(IntegrationCategory.LLM_GUARDRAIL,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_GUARDRAILS_AI",
    description="Guardrails AI LLM guardrail harness adapter (M.12)",
)
