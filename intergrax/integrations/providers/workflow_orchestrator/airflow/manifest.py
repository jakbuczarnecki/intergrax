# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``airflow`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="airflow",
    categories=(IntegrationCategory.WORKFLOW_ORCHESTRATOR,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_AIRFLOW',
    description='airflow integration (Phase M.6 P6)',
)
