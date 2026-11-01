# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from testing_support.change_safety.assess.consumers import count_critical_consumer_omissions
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory


@dataclass(frozen=True)
class GraphCoverageReport:
    schema_version: str
    module_nodes: int
    modules_with_import_edges: int
    coverage_ratio: float
    critical_omissions: int


def compute_graph_coverage(inventory: ClosedWorldInventory) -> GraphCoverageReport:
    module_nodes = {n.node_id for n in inventory.nodes if n.node_id.startswith("module:")}
    sources = {e.source_id for e in inventory.edges if e.rel == "IMPORTS"}
    covered = len(module_nodes & sources)
    total = len(module_nodes)
    ratio = covered / total if total else 0.0
    omissions = count_critical_consumer_omissions(inventory)
    return GraphCoverageReport(
        schema_version="evol_graph_coverage.v1",
        module_nodes=total,
        modules_with_import_edges=covered,
        coverage_ratio=ratio,
        critical_omissions=omissions,
    )
