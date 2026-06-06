# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``replicate`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="replicate",
    categories=(IntegrationCategory.ML_INFERENCE_HOST,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_REPLICATE',
    description='replicate integration (Phase M.6 P6)',
)
