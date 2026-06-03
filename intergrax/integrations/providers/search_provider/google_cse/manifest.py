# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``google_cse`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="google_cse",
    categories=(IntegrationCategory.SEARCH_PROVIDER,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_GOOGLE_CSE',
    description='Google Custom Search — web search via REST API (via create_google_cse_integration; legacy GOOGLE_CSE_* env supported)',
)
