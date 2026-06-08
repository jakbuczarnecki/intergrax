# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``bigquery`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="bigquery",
    categories=(IntegrationCategory.RELATIONAL_STORE,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_BIGQUERY',
    description='bigquery integration (Phase M.7 P7)',
)
