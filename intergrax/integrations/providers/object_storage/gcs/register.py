# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register gcs in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.object_storage.gcs.bundle import create_gcs_object_storage
from intergrax.integrations.providers.object_storage.gcs.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_gcs_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_gcs_object_storage, override=override)
