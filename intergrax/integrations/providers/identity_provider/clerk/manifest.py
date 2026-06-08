# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``clerk`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="clerk",
    categories=(IntegrationCategory.IDENTITY_PROVIDER,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_CLERK',
    description='clerk integration (Phase M.7 P7)',
)
