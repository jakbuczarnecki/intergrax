# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register google_drive in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.object_storage.google_drive.bundle import create_google_drive_object_storage
from intergrax.integrations.providers.object_storage.google_drive.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_google_drive_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_google_drive_object_storage, override=override)
