# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``dynamodb`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="dynamodb",
    categories=(IntegrationCategory.DOCUMENT_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_DYNAMODB',
    description='dynamodb integration (Phase M.6 P2/P3)',
)
