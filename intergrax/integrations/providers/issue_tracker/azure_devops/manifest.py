# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``azure_devops`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="azure_devops",
    categories=(IntegrationCategory.ISSUE_TRACKER,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_AZURE_DEVOPS',
    description='azure_devops integration (Phase M.6 P2/P3)',
)
