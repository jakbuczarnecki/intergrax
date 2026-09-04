# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.integrations.providers.graph_store.arangodb.bundle import create_arangodb_graph_store
from intergrax.integrations.providers.graph_store.arangodb.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.graph_store.arangodb.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_arangodb_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_arangodb_graph_store, override=override, contract_specs=CONTRACT_SPECS)
