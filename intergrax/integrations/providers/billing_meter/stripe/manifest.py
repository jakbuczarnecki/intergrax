# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``stripe`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="stripe",
    categories=(IntegrationCategory.BILLING_METER,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_STRIPE',
    description='stripe integration (Phase M.6 P6)',
)
