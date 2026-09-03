# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register azure_blob in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.object_storage.azure_blob.bundle import create_azure_blob_object_storage
from intergrax.integrations.providers.object_storage.azure_blob.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.providers.object_storage.azure_blob.contract_spec import CONTRACT_SPECS


def register_azure_blob_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_azure_blob_object_storage, override=override, contract_specs=CONTRACT_SPECS)
