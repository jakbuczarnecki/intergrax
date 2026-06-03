# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``gcp`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="gcp",
    categories=(IntegrationCategory.CLOUD_PLATFORM,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_GCP',
    description='GCP cloud platform facade (ADC / service account; defaults for GCS, Pub/Sub, Cloud SQL)',
)
