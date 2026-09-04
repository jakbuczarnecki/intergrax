# © Artur Czarnecki. All rights reserved.

"""Catalog manifest for ``llama_guard`` llm_guardrail integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="llama_guard",
    categories=(IntegrationCategory.LLM_GUARDRAIL,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_LLAMA_GUARD",
    description="Llama Guard LLM guardrail harness adapter (M.12)",
)
