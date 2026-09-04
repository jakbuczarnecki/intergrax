# © Artur Czarnecki. All rights reserved.

"""Catalog manifest for ``nemo_guardrails`` llm_guardrail integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="nemo_guardrails",
    categories=(IntegrationCategory.LLM_GUARDRAIL,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_NEMO_GUARDRAILS",
    description="NeMo Guardrails LLM guardrail harness adapter (M.12)",
)
