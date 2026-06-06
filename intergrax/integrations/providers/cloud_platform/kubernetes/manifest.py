# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``kubernetes`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="kubernetes",
    categories=(IntegrationCategory.CLOUD_PLATFORM,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_KUBERNETES',
    description='kubernetes integration (Phase M.6 P4)',
)
