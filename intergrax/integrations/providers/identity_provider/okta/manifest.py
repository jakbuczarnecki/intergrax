# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``okta`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="okta",
    categories=(IntegrationCategory.IDENTITY_PROVIDER,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_OKTA',
    description='okta integration (Phase M.7 P7)',
)
