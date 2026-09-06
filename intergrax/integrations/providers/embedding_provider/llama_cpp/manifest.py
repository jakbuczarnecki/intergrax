# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``llama_cpp`` embedding provider integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="llama_cpp",
    categories=(IntegrationCategory.EMBEDDING_PROVIDER,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_LLAMA_CPP_EMBEDDING",
    description="llama.cpp OpenAI-compatible embedding server provider.",
)
