# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``docling`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="docling",
    categories=(IntegrationCategory.DOCUMENT_PARSER,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_DOCLING',
    description='Docling document parser — local library or HTTP server',
)
