# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``azure_blob`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="azure_blob",
    categories=(IntegrationCategory.OBJECT_STORAGE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_AZURE_BLOB',
    description='Azure Blob Storage object storage (put/get/delete/presigned_url)',
)
