# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``infisical`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="infisical",
    categories=(IntegrationCategory.SECRETS_STORE,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_INFISICAL',
    description='infisical integration (Phase M.6 P6)',
)
