# © Artur Czarnecki. All rights reserved.

"""Catalog manifest for ``lakera`` llm_guardrail integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="lakera",
    categories=(IntegrationCategory.LLM_GUARDRAIL,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_LAKERA",
    description="Lakera LLM guardrail harness adapter (M.12)",
)
