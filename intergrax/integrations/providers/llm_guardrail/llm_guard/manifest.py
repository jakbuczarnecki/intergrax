# © Artur Czarnecki. All rights reserved.

"""Catalog manifest for ``llm_guard`` llm_guardrail integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="llm_guard",
    categories=(IntegrationCategory.LLM_GUARDRAIL,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_LLM_GUARD",
    description="LLM Guard LLM guardrail harness adapter (M.12)",
)
