# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from testing_support.change_safety.contracts.enums import ImpactStatus
from testing_support.change_safety.contracts.impact import AffectedNode, ImpactAssessmentResult
from testing_support.change_safety.data.loaders import load_evol_x_data_document
from testing_support.change_safety.discovery.closed_world import (
    ClosedWorldInventory,
    accepted_exclusion_patterns,
    path_matches_patterns,
)
from testing_support.change_safety.discovery.extractors import GraphEdge, module_id_for_path, suite_id_for_path
from testing_support.change_safety.graph.builder import build_semantic_graph
from testing_support.change_safety.graph.traversal import traverse_from_changed_modules


def _config_nodes_for_path(inventory: ClosedWorldInventory, path: str) -> tuple[str, ...]:
    base = path.rsplit("/", 1)[-1]
    out = []
    for node in inventory.nodes:
        if node.category == "config" and not node.node_id.startswith("config:env:"):
            literal = node.node_id[len("config:") :]
            if path.endswith(literal) or literal.endswith(base):
                out.append(node.node_id)
    return tuple(out)


def assess_path_change_impact(
    inventory: ClosedWorldInventory,
    *,
    changed_paths: frozenset[str],
) -> ImpactAssessmentResult:
    graph_digest = build_semantic_graph(inventory).digest
    if not changed_paths:
        return ImpactAssessmentResult(
            source_commit_sha=inventory.source_commit_sha,
            graph_digest=graph_digest,
            inventory_digest=inventory.inventory_digest,
            ruleset_version=inventory.ruleset_version,
            status=ImpactStatus.COMPLETE,
            affected=(),
        )
    exclusions = accepted_exclusion_patterns(
        tuple(load_evol_x_data_document("exclusion_register.v1.json").get("entries", ()))
    )
    module_ids: set[str] = set()
    extra: dict[str, AffectedNode] = {}
    unknown_paths: list[str] = []
    suites = {n.node_id for n in inventory.nodes if n.category == "suite"}
    for path in sorted(changed_paths):
        normalized = path.replace("\\", "/")
        if normalized.endswith(".py"):
            mid = module_id_for_path(normalized)
            sid = suite_id_for_path(normalized)
            if mid in inventory.module_ids:
                module_ids.add(mid)
            elif sid in suites:
                extra[sid] = AffectedNode(node_id=sid, kind="DIRECT", witness_path=(sid,), rule_id="R-SUITE-DIRECT")
            elif path_matches_patterns(normalized, exclusions):
                extra[f"excluded:{normalized}"] = AffectedNode(
                    node_id=f"excluded:{normalized}", kind="EXCLUDED", witness_path=(normalized,), rule_id="R-EXCLUDED"
                )
            else:
                unknown_paths.append(normalized)
        elif path_matches_patterns(normalized, exclusions):
            extra[f"excluded:{normalized}"] = AffectedNode(
                node_id=f"excluded:{normalized}", kind="EXCLUDED", witness_path=(normalized,), rule_id="R-EXCLUDED"
            )
        else:
            configs = _config_nodes_for_path(inventory, normalized)
            if configs:
                for cfg in configs:
                    extra[cfg] = AffectedNode(
                        node_id=cfg, kind="DIRECT", witness_path=(normalized,), rule_id="R-CONFIG-BINDS"
                    )
                    module_ids.update(src for src, _rel in inventory.consumers_reverse.get(cfg, ()))
            else:
                unknown_paths.append(normalized)
    affected_map = traverse_from_changed_modules(inventory, frozenset(module_ids))
    affected_map.update({k: v for k, v in extra.items() if k not in affected_map})
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
        graph_digest=graph_digest,
        inventory_digest=inventory.inventory_digest,
        ruleset_version=inventory.ruleset_version,
        status=status,
        affected=tuple(sorted(affected_map.values(), key=lambda a: a.node_id)),
    )


def edges_by_rel(edges: tuple[GraphEdge, ...], rel: str) -> tuple[GraphEdge, ...]:
    return tuple(e for e in edges if e.rel == rel)
