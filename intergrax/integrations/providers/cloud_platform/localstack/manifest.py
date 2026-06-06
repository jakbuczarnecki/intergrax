# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``localstack`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="localstack",
    categories=(IntegrationCategory.CLOUD_PLATFORM,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_LOCALSTACK',
    description='localstack integration (Phase M.6 P5)',
)
