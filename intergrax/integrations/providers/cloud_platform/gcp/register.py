# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register GCP in the integration catalog (Phase M.6)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.cloud_platform.gcp.bundle import create_gcp_cloud_platform
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_gcp_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.GCP.value,
            categories=(IntegrationCategory.CLOUD_PLATFORM,),
            factory=create_gcp_cloud_platform,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_GCP",
            description=(
                "GCP cloud platform facade (ADC / service account; defaults for GCS, Pub/Sub, Cloud SQL)"
            ),
        ),
        override=override,
    )
