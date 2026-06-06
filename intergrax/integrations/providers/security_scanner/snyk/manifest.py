# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``snyk`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="snyk",
    categories=(IntegrationCategory.SECURITY_SCANNER,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_SNYK',
    description='snyk integration (Phase M.6 P6)',
)
