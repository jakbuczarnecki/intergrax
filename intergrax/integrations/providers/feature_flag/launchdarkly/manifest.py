# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``launchdarkly`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="launchdarkly",
    categories=(IntegrationCategory.FEATURE_FLAG,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_LAUNCHDARKLY',
    description='SaaS-only feature flag backend (Phase M.6 P4)',
    requires_local_container=False,
)
