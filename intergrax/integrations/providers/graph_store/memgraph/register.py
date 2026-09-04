# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register memgraph in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.graph_store.memgraph.bundle import create_memgraph_graph_store
from intergrax.integrations.providers.graph_store.memgraph.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.providers.graph_store.memgraph.contract_spec import CONTRACT_SPECS


def register_memgraph_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_memgraph_graph_store, override=override, contract_specs=CONTRACT_SPECS)
