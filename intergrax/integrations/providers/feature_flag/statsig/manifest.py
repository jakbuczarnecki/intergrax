# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``statsig`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="statsig",
    categories=(IntegrationCategory.FEATURE_FLAG,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_STATSIG',
    description='statsig integration (Phase M.6 P6)',
)
