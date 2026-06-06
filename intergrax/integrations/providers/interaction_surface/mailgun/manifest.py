# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``mailgun`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="mailgun",
    categories=(IntegrationCategory.INTERACTION_SURFACE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_MAILGUN',
    description='mailgun integration (Phase M.6 P4)',
)
