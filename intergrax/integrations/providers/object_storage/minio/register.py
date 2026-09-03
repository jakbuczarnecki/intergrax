# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register minio in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.object_storage.minio.bundle import create_minio_object_storage
from intergrax.integrations.providers.object_storage.minio.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.providers.object_storage.minio.contract_spec import CONTRACT_SPECS


def register_minio_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_minio_object_storage, override=override, contract_specs=CONTRACT_SPECS)
