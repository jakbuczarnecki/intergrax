# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``openai`` managed retrieval integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="openai",
    categories=(IntegrationCategory.MANAGED_RETRIEVAL,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_OPENAI",
    description="OpenAI hosted managed retrieval (vector stores + file_search).",
)
