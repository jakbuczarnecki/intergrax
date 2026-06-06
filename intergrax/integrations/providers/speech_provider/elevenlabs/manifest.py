# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``elevenlabs`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="elevenlabs",
    categories=(IntegrationCategory.SPEECH_PROVIDER,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_ELEVENLABS',
    description='elevenlabs integration (Phase M.6 P6)',
)
