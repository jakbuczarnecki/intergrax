# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register filesystem in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.object_storage.filesystem.bundle import create_filesystem_object_storage
from intergrax.integrations.providers.object_storage.filesystem.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_filesystem_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_filesystem_object_storage, override=override)
