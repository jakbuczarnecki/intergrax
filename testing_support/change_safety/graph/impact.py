# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from testing_support.change_safety.contracts.enums import ImpactStatus
from testing_support.change_safety.contracts.impact import AffectedNode, ImpactAssessmentResult
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.discovery.extractors import GraphEdge
from testing_support.change_safety.graph.traversal import traverse_from_changed_modules


def assess_path_change_impact(
    inventory: ClosedWorldInventory,
    *,
    changed_paths: frozenset[str],
) -> ImpactAssessmentResult:
    if not changed_paths:
        return ImpactAssessmentResult(
            source_commit_sha=inventory.source_commit_sha,
            graph_digest=inventory.inventory_digest,
            inventory_digest=inventory.inventory_digest,
            ruleset_version=inventory.ruleset_version,
            status=ImpactStatus.COMPLETE,
            affected=(),
        )
    module_ids: set[str] = set()
    unknown_paths: list[str] = []
    for path in sorted(changed_paths):
        normalized = path.replace("\\", "/")
        if normalized.endswith(".py"):
            module_ids.add(f"module:{normalized[:-3].replace('/', '.')}")
        else:
            unknown_paths.append(normalized)
    affected_map = traverse_from_changed_modules(inventory.edges, frozenset(module_ids))
    for path in unknown_paths:
        affected_map[f"unknown:path:{path}"] = AffectedNode(
            node_id=f"unknown:path:{path}",
            kind="CONSERVATIVE_UNKNOWN",
            witness_path=(path,),
            rule_id="R-UNKNOWN-PATH",
        )
    status = ImpactStatus.COMPLETE if not unknown_paths else ImpactStatus.PARTIAL
    return ImpactAssessmentResult(
        source_commit_sha=inventory.source_commit_sha,
        graph_digest=inventory.inventory_digest,
        inventory_digest=inventory.inventory_digest,
        ruleset_version=inventory.ruleset_version,
        status=status,
        affected=tuple(sorted(affected_map.values(), key=lambda a: a.node_id)),
    )


def edges_by_rel(edges: tuple[GraphEdge, ...], rel: str) -> tuple[GraphEdge, ...]:
    return tuple(e for e in edges if e.rel == rel)
