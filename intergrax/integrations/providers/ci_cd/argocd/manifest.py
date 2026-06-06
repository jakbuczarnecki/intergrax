# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``argocd`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="argocd",
    categories=(IntegrationCategory.CI_CD,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_ARGOCD',
    description='argocd integration (Phase M.6 P6)',
)
