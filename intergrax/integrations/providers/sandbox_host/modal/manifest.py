# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``modal`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="modal",
    categories=(IntegrationCategory.SANDBOX_HOST,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_MODAL',
    description='modal integration (Phase M.6 P6)',
)
