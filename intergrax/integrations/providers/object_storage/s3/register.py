# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register s3 in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.object_storage.s3.bundle import create_s3_object_storage
from intergrax.integrations.providers.object_storage.s3.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.providers.object_storage.s3.contract_spec import CONTRACT_SPECS


def register_s3_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_s3_object_storage, override=override, contract_specs=CONTRACT_SPECS)
