# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``vllm`` embedding provider integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="vllm",
    categories=(IntegrationCategory.EMBEDDING_PROVIDER,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_VLLM_EMBEDDING",
    description="vLLM OpenAI-compatible embedding server provider.",
)
