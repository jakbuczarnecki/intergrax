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
    import_ratio_internal: float = 0.0


def compute_graph_coverage(inventory: ClosedWorldInventory) -> GraphCoverageReport:
    module_nodes = set(inventory.module_ids)
    sources = {e.source_id for e in inventory.edges if e.rel in {"IMPORTS", "CONSUMES_CONTRACT", "BINDS", "IMPLEMENTS"}}
    covered = len(module_nodes & sources)
    total = len(module_nodes)
    ratio = covered / total if total else 0.0
    stats = dict(inventory.import_stats)
    total_imports = stats.get("total_import_records", 0)
    internal_ratio = stats.get("internal_import_records", 0) / total_imports if total_imports else 0.0
    return GraphCoverageReport(
        schema_version="evol_graph_coverage.v1",
        module_nodes=total,
        modules_with_import_edges=covered,
        coverage_ratio=ratio,
        critical_omissions=count_critical_consumer_omissions(inventory),
        import_ratio_internal=internal_ratio,
    )
