# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``confluent`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="confluent",
    categories=(IntegrationCategory.MESSAGE_BUS,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_CONFLUENT',
    description='confluent integration (Phase M.6 P6)',
)
