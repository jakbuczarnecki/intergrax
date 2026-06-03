# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register mongodb in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.document_store.mongodb.bundle import create_mongodb_document_store
from intergrax.integrations.providers.document_store.mongodb.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_mongodb_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_mongodb_document_store, override=override)
