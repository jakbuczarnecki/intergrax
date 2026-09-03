# © Artur Czarnecki. All rights reserved.

"""Register H-INT-GRAPH graph_store expansion slugs (Neptune, OrientDB, ArangoDB)."""

from __future__ import annotations


def register_h_int_graph_integrations(*, override: bool = False) -> None:
    """Defer H-INT-GRAPH catalog rows until explicit contract_specs land (P2-003-B1 gate)."""
    _ = override
