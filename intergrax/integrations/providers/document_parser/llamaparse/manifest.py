# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``llamaparse`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="llamaparse",
    categories=(IntegrationCategory.DOCUMENT_PARSER,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_LLAMAPARSE',
    description='llamaparse integration (Phase M.7 P7)',
)
