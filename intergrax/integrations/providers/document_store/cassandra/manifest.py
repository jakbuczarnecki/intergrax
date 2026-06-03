# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``cassandra`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="cassandra",
    categories=(IntegrationCategory.DOCUMENT_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_CASSANDRA',
    description='Cassandra wide-column document store (partition-scoped get/put/delete/query)',
)
