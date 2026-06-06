# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``gcp_secret_manager`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="gcp_secret_manager",
    categories=(IntegrationCategory.SECRETS_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_GCP_SECRET_MANAGER',
    description='gcp_secret_manager integration (Phase M.6 P4)',
)
