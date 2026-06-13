# © Artur Czarnecki. All rights reserved.

"""Register H-INT-GRAPH graph_store expansion slugs (Neptune, OrientDB, ArangoDB)."""

from __future__ import annotations


def register_h_int_graph_integrations(*, override: bool = False) -> None:
    from intergrax.integrations.providers.graph_store.neptune.register import register_neptune_integration
    from intergrax.integrations.providers.graph_store.orientdb.register import register_orientdb_integration
    from intergrax.integrations.providers.graph_store.arangodb.register import register_arangodb_integration

    register_neptune_integration(override=override)
    register_orientdb_integration(override=override)
    register_arangodb_integration(override=override)
