# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``auth0`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="auth0",
    categories=(IntegrationCategory.IDENTITY_PROVIDER,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_AUTH0',
    description='auth0 integration (Phase M.6 P6)',
)
