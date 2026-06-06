# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``mongodb`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="mongodb",
    categories=(IntegrationCategory.DOCUMENT_STORE,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_MONGODB',
    description='MongoDB flexible document store (partition-scoped get/put/delete/query)',
)
