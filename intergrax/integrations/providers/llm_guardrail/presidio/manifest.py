# © Artur Czarnecki. All rights reserved.

"""Catalog manifest for ``presidio`` llm_guardrail integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="presidio",
    categories=(IntegrationCategory.LLM_GUARDRAIL,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_PRESIDIO",
    description="Presidio LLM guardrail harness adapter (M.12)",
)
