# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``airbyte`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="airbyte",
    categories=(IntegrationCategory.WORKFLOW_ORCHESTRATOR,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_AIRBYTE',
    description='airbyte integration (Phase M.7 P7)',
)
