# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``vespa`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="vespa",
    categories=(IntegrationCategory.VECTOR_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_VESPA',
    description='vespa integration (Phase M.8 harness)',
)
