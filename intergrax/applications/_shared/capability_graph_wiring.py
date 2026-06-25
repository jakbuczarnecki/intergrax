# © Artur Czarnecki. All rights reserved.

"""Tier-3 environment capability graph wiring (Phase CG-1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdge,
    CapabilityEdgeType,
    CapabilityGraph,
    CapabilityNode,
    CapabilityNodeType,
)
from intergrax.runtime.architecture.capability_graph_applications import (
    application_capability_node_id,
    resolve_binding_agent_contract_id,
)


@dataclass(frozen=True, slots=True)
class EnvironmentCapabilityGraphView:
    """Environment-scoped slice of a capability graph."""

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


def _agent_contract_from_binding(binding: AgentBinding) -> object | None:
    if binding.agent_type is None and binding.import_path is None:
        return None
    return binding.resolved_agent_type()().get_contract()


def build_environment_seed_capability_graph(
    manifest: ApplicationManifest,
    snapshot: HarnessRegistrySnapshot,
) -> CapabilityGraph:
    """Build an environment-local graph without importing unrelated reference agents.

    Product hosts must not import global demo/reference registries while starting. This
    graph is intentionally seeded from the application manifest and the registries that
    were actually wired for this environment.
    """
    node_by_id: dict[str, CapabilityNode] = {}
    edges: list[CapabilityEdge] = []

    for node_id in sorted(_seed_node_ids(manifest, snapshot)):
        node_by_id[node_id] = CapabilityNode(node_id=node_id, node_type=_node_type_from_id(node_id))

    application_node = application_capability_node_id(manifest)
    policy_node = "policy:runtime_policy_bundle"
    edges.append(
        CapabilityEdge(
            source_node_id=application_node,
            target_node_id=policy_node,
            edge_type=CapabilityEdgeType.CONSTRAINED_BY,
        )
    )

    for binding in manifest.enabled_agents():
        contract_id = resolve_binding_agent_contract_id(binding)
        agent_node = f"agent:{contract_id}"
        node_by_id.setdefault(agent_node, CapabilityNode(node_id=agent_node, node_type=CapabilityNodeType.AGENT))
        edges.append(
            CapabilityEdge(
                source_node_id=application_node,
                target_node_id=agent_node,
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            )
        )
        contract = _agent_contract_from_binding(binding)
        if contract is None:
            continue
        for skill_manifest in getattr(contract, "skills", ()):  # pragma: no cover - defensive for legacy contracts
            skill_id = getattr(skill_manifest, "skill_id", None)
            if not skill_id:
                continue
            skill_node = f"skill:{skill_id}"
            node_by_id.setdefault(skill_node, CapabilityNode(node_id=skill_node, node_type=CapabilityNodeType.SKILL))
            edges.append(
                CapabilityEdge(
                    source_node_id=agent_node,
                    target_node_id=skill_node,
                    edge_type=CapabilityEdgeType.DEPENDS_ON,
                )
            )
        for tool_id in getattr(contract, "allowed_tools", ()):  # pragma: no cover - defensive for legacy contracts
            tool_node = f"tool:{tool_id}"
            node_by_id.setdefault(tool_node, CapabilityNode(node_id=tool_node, node_type=CapabilityNodeType.TOOL))
            edges.append(
                CapabilityEdge(
                    source_node_id=agent_node,
                    target_node_id=tool_node,
                    edge_type=CapabilityEdgeType.DEPENDS_ON,
                )
            )

    return CapabilityGraph(
        nodes=sorted(node_by_id.values(), key=lambda item: item.node_id),
        edges=edges,
    )


def extract_environment_capability_graph(
    catalog: CapabilityGraph,
    *,
    seed_node_ids: frozenset[str],
) -> CapabilityGraph:
    """Return connected subgraph containing seeds and catalog neighbors."""
    node_by_id = {node.node_id: node for node in catalog.nodes}
    included: set[str] = set(seed_node_ids)

    for seed in seed_node_ids:
        if seed not in node_by_id:
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
    """Materialize an environment-local capability graph from wired registries."""
    _ = env
    if catalog is None:
        graph = build_environment_seed_capability_graph(manifest, snapshot)
    else:
        seeds = _seed_node_ids(manifest, snapshot)
        graph = extract_environment_capability_graph(catalog, seed_node_ids=seeds)
    return EnvironmentCapabilityGraphView(graph=graph)


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
