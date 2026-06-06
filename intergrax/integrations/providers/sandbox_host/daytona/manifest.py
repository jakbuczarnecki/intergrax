# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``daytona`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="daytona",
    categories=(IntegrationCategory.SANDBOX_HOST,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_DAYTONA',
    description='daytona integration (Phase M.6 P6)',
)
