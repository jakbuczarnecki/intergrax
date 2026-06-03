# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``unstructured`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="unstructured",
    categories=(IntegrationCategory.DOCUMENT_PARSER,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_UNSTRUCTURED',
    description='Unstructured HTML document parser',
)
