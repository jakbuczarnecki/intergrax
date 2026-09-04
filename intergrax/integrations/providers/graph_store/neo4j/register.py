# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register neo4j in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.graph_store.neo4j.bundle import create_neo4j_graph_store
from intergrax.integrations.providers.graph_store.neo4j.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.providers.graph_store.neo4j.contract_spec import CONTRACT_SPECS


def register_neo4j_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_neo4j_graph_store, override=override, contract_specs=CONTRACT_SPECS)
