# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``tempo`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="tempo",
    categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_TEMPO',
    description='tempo integration (Phase M.6 P4)',
)
