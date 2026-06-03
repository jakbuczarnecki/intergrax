# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register cassandra in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.document_store.cassandra.bundle import create_cassandra_document_store
from intergrax.integrations.providers.document_store.cassandra.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_cassandra_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_cassandra_document_store, override=override)
