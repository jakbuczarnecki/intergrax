# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register huggingface_hub in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.object_storage.huggingface_hub.bundle import create_huggingface_hub_object_storage
from intergrax.integrations.providers.object_storage.huggingface_hub.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_huggingface_hub_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_huggingface_hub_object_storage, override=override)
