# © Artur Czarnecki. All rights reserved.

"""Tier-3 environment capability graph wiring (Phase CG-1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.architecture.capability_graph import (
    CapabilityGraph,
    CapabilityNode,
    CapabilityNodeType,
    build_catalog_capability_graph,
)
from intergrax.runtime.architecture.capability_graph_applications import (
    application_capability_node_id,
    resolve_binding_agent_contract_id,
)


@dataclass(frozen=True, slots=True)
class EnvironmentCapabilityGraphView:
    """Environment-scoped slice of the catalog capability graph."""

    graph: CapabilityGraph

    def node_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.graph.nodes)

    def contains_node(self, node_id: str) -> bool:
        return any(node.node_id == node_id for node in self.graph.nodes)


def _seed_node_ids(
    manifest: ApplicationManifest,
    snapshot: HarnessRegistrySnapshot,
) -> frozenset[str]:
    seeds: set[str] = {application_capability_node_id(manifest), "policy:runtime_policy_bundle"}
    for tool_id in snapshot.tool_ids():
        seeds.add(f"tool:{tool_id}")
    for skill_id in snapshot.skill_ids():
        seeds.add(f"skill:{skill_id}")
    for prompt_id in snapshot.prompt_ids():
        seeds.add(f"prompt:{prompt_id}")
    for binding in manifest.enabled_agents():
        contract_id = resolve_binding_agent_contract_id(binding)
        seeds.add(f"agent:{contract_id}")
    return frozenset(seeds)


def _node_type_from_id(node_id: str) -> CapabilityNodeType:
    prefix = node_id.split(":", maxsplit=1)[0]
    mapping = {
        "integration": CapabilityNodeType.INTEGRATION,
        "tool": CapabilityNodeType.TOOL,
        "skill": CapabilityNodeType.SKILL,
        "policy": CapabilityNodeType.POLICY,
        "prompt": CapabilityNodeType.PROMPT,
        "agent": CapabilityNodeType.AGENT,
        "application": CapabilityNodeType.APPLICATION,
        "product": CapabilityNodeType.PRODUCT,
        "evaluation": CapabilityNodeType.EVALUATION,
    }
    try:
        return mapping[prefix]
    except KeyError as exc:
        raise ValueError(f"Unknown capability node prefix in {node_id!r}") from exc


def _synthetic_seed_prefixes() -> frozenset[str]:
    return frozenset({"application", "agent", "policy"})


def extract_environment_capability_graph(
    catalog: CapabilityGraph,
    *,
    seed_node_ids: frozenset[str],
) -> CapabilityGraph:
    """Return connected subgraph containing seeds and catalog neighbors."""
    node_by_id = {node.node_id: node for node in catalog.nodes}
    included: set[str] = set(seed_node_ids)

    for seed in seed_node_ids:
        prefix = seed.split(":", maxsplit=1)[0]
        if prefix in _synthetic_seed_prefixes() and seed not in node_by_id:
            node_by_id[seed] = CapabilityNode(node_id=seed, node_type=_node_type_from_id(seed))

    changed = True
    while changed:
        changed = False
        for edge in catalog.edges:
            if edge.source_node_id in included or edge.target_node_id in included:
                if edge.source_node_id not in included and edge.source_node_id in node_by_id:
                    included.add(edge.source_node_id)
                    changed = True
                if edge.target_node_id not in included and edge.target_node_id in node_by_id:
                    included.add(edge.target_node_id)
                    changed = True

    nodes = [node_by_id[node_id] for node_id in sorted(included) if node_id in node_by_id]
    edges = [
        edge
        for edge in catalog.edges
        if edge.source_node_id in included and edge.target_node_id in included
    ]
    return CapabilityGraph(nodes=nodes, edges=edges)


def resolve_environment_capability_graph(
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
    snapshot: HarnessRegistrySnapshot,
    *,
    catalog: CapabilityGraph | None = None,
) -> EnvironmentCapabilityGraphView:
    """Materialize environment capability graph from catalog baseline and wired registries."""
    _ = env
    catalog_graph = catalog or build_catalog_capability_graph()
    seeds = _seed_node_ids(manifest, snapshot)
    subgraph = extract_environment_capability_graph(catalog_graph, seed_node_ids=seeds)
    return EnvironmentCapabilityGraphView(graph=subgraph)


def wire_environment_capability_graph(
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
    snapshot: HarnessRegistrySnapshot,
    *,
    catalog: CapabilityGraph | None = None,
) -> EnvironmentCapabilityGraphView:
    """Alias for :func:`resolve_environment_capability_graph` (architecture §50.1.2)."""
    return resolve_environment_capability_graph(
        manifest,
        env,
        snapshot,
        catalog=catalog,
    )
