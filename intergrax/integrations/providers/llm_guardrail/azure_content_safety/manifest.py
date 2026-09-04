# © Artur Czarnecki. All rights reserved.

"""Catalog manifest for ``azure_content_safety`` llm_guardrail integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="azure_content_safety",
    categories=(IntegrationCategory.LLM_GUARDRAIL,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_AZURE_CONTENT_SAFETY",
    description="Azure Content Safety LLM guardrail harness adapter (M.12)",
)
