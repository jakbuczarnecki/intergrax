# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``azure`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="azure",
    categories=(IntegrationCategory.CLOUD_PLATFORM,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_AZURE',
    description='Azure cloud platform facade (MI / service principal; defaults for Blob, Service Bus, Azure SQL)',
)
