# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``lab_json`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="lab_json",
    categories=(IntegrationCategory.INTERACTION_SURFACE,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_LAB_JSON',
    description='Laboratory JSON interaction surface — vendor-neutral dict → Task (via create_lab_json_integration)',
)
