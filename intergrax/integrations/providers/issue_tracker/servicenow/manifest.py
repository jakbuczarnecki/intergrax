# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``servicenow`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="servicenow",
    categories=(IntegrationCategory.ISSUE_TRACKER,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_SERVICENOW',
    description='servicenow integration (Phase M.6 P4)',
)
