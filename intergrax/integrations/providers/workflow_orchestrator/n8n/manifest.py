# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``n8n`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="n8n",
    categories=(IntegrationCategory.WORKFLOW_ORCHESTRATOR,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_N8N',
    description='n8n integration (Phase M.7 P7)',
)
