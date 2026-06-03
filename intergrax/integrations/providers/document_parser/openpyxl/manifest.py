# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``openpyxl`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="openpyxl",
    categories=(IntegrationCategory.DOCUMENT_PARSER,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_OPENPYXL',
    description='Excel/CSV parsing via pandas and openpyxl',
)
