# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``unleash`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="unleash",
    categories=(IntegrationCategory.FEATURE_FLAG,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_UNLEASH',
    description='unleash integration (Phase M.6 P4)',
)
