# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``doppler`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="doppler",
    categories=(IntegrationCategory.SECRETS_STORE,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_DOPPLER',
    description='doppler integration (Phase M.6 P4)',
)
