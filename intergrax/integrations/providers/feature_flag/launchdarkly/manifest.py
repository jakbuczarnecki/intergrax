# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``launchdarkly`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="launchdarkly",
    categories=(IntegrationCategory.FEATURE_FLAG,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_LAUNCHDARKLY',
    description='launchdarkly integration (Phase M.6 P4)',
)
