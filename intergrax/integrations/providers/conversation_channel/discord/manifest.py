# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``discord`` conversation channel integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="discord",
    categories=(IntegrationCategory.CONVERSATION_CHANNEL,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_DISCORD",
    description="Discord conversation channel (contract-defined, runtime-unbound)",
)
