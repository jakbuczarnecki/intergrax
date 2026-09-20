# © Artur Czarnecki. All rights reserved.

"""Synthetic manifest for X5A anti-drift gate tests (not part of production catalog)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="x5a_unclassified_fake",
    categories=(IntegrationCategory.DOCUMENT_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_X5A_UNCLASSIFIED_FAKE",
    description="X5A anti-drift fixture — must fail classification gate when discovered",
)
