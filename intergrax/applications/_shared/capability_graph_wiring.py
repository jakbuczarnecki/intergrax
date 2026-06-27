# © Artur Czarnecki. All rights reserved.

"""Tier-3 environment capability graph wiring (Phase CG-1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdge,
    CapabilityEdgeType,
    CapabilityGraph,
    CapabilityNode,
    CapabilityNodeType,
)
from intergrax.applications._shared.capability_graph_catalog import resolve_binding_agent_contract_id
from intergrax.runtime.architecture.capability_graph_applications import application_capability_node_id


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
    seeds: set[str] = {application_capability_node_id(manifest.app_id), "policy:runtime_policy_bundle"}
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


def _agent_contract_from_binding(binding: AgentBinding) -> AgentContract | None:
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

    application_node = application_capability_node_id(manifest.app_id)
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
        for skill_manifest in contract.skills:
            skill_id = skill_manifest.skill_id
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
        for tool_id in contract.allowed_tools:
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


def build_environment_capability_graph_from_wiring(
    manifest: ApplicationManifest,
    snapshot: HarnessRegistrySnapshot,
) -> CapabilityGraph:
    """Build capability graph from manifest roster and environment registry snapshot only."""
    application_node = application_capability_node_id(manifest.app_id)
    nodes_by_id: dict[str, CapabilityNode] = {}
    node_ids: set[str] = set()
    edges: list[CapabilityEdge] = []

    def ensure_node(node_id: str, node_type: CapabilityNodeType, version: str | None = None) -> None:
        if node_id not in node_ids:
            node_kwargs: dict[str, object] = {"node_id": node_id, "node_type": node_type}
            if version is not None:
                node_kwargs["version"] = version
            nodes_by_id[node_id] = CapabilityNode(**node_kwargs)  # type: ignore[arg-type]
            node_ids.add(node_id)

    def add_edge(source: str, target: str, edge_type: CapabilityEdgeType) -> None:
        if source in node_ids and target in node_ids:
            edges.append(
                CapabilityEdge(
                    source_node_id=source,
                    target_node_id=target,
                    edge_type=edge_type,
                )
            )

    ensure_node(application_node, CapabilityNodeType.APPLICATION)
    ensure_node("policy:runtime_policy_bundle", CapabilityNodeType.POLICY)
    add_edge(application_node, "policy:runtime_policy_bundle", CapabilityEdgeType.CONSTRAINED_BY)

    for tool_id in snapshot.tool_ids():
        ensure_node(f"tool:{tool_id}", CapabilityNodeType.TOOL)
    for skill_id in snapshot.skill_ids():
        ensure_node(f"skill:{skill_id}", CapabilityNodeType.SKILL)
    for prompt_id in snapshot.prompt_ids():
        ensure_node(f"prompt:{prompt_id}", CapabilityNodeType.PROMPT)

    if snapshot.skill_registry is not None:
        for registered in snapshot.skill_registry.list():
            skill_node = f"skill:{registered.manifest.skill_id}"
            if skill_node not in node_ids:
                continue
            for tool_id in registered.manifest.tool_ids:
                tool_node = f"tool:{tool_id}"
                add_edge(skill_node, tool_node, CapabilityEdgeType.DEPENDS_ON)

    agent_registry = snapshot.agent_registry
    contract_versions: dict[str, str | None] = {}
    if agent_registry is not None:
        for contract in agent_registry.list_contracts():
            contract_versions[contract.id] = contract.version

    for binding in manifest.enabled_agents():
        contract_id = resolve_binding_agent_contract_id(binding)
        agent_node = f"agent:{contract_id}"
        ensure_node(agent_node, CapabilityNodeType.AGENT, version=contract_versions.get(contract_id))
        add_edge(application_node, agent_node, CapabilityEdgeType.DEPENDS_ON)

    if agent_registry is not None:
        for contract in agent_registry.list_contracts():
            agent_node = f"agent:{contract.id}"
            ensure_node(agent_node, CapabilityNodeType.AGENT, version=contract.version)
            for skill_manifest in contract.skills:
                add_edge(agent_node, f"skill:{skill_manifest.skill_id}", CapabilityEdgeType.DEPENDS_ON)
            for tool_id in contract.allowed_tools:
                add_edge(agent_node, f"tool:{tool_id}", CapabilityEdgeType.DEPENDS_ON)
            if contract.prompt_binding_id:
                add_edge(agent_node, f"prompt:{contract.prompt_binding_id}", CapabilityEdgeType.DEPENDS_ON)

    for eval_id in snapshot.evaluation_registry_ids():
        ensure_node(eval_id, CapabilityNodeType.EVALUATION)
        for node_id in sorted(node_ids):
            if node_id.startswith("agent:"):
                add_edge(eval_id, node_id, CapabilityEdgeType.EVALUATES)

    return CapabilityGraph(
        nodes=[nodes_by_id[node_id] for node_id in sorted(nodes_by_id)],
        edges=edges,
    )


def resolve_environment_capability_graph(
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
    snapshot: HarnessRegistrySnapshot,
    *,
    catalog: CapabilityGraph | None = None,
) -> EnvironmentCapabilityGraphView:
    """Materialize environment capability graph from wired registries.

    When ``catalog`` is omitted, the graph is built from the application manifest,
    environment registry snapshot, and roster agents only. Pass ``catalog=...`` to
    slice an explicit global catalog baseline (governance / reference tooling).
    """
    _ = env
    if catalog is not None:
        seeds = _seed_node_ids(manifest, snapshot)
        subgraph = extract_environment_capability_graph(catalog, seed_node_ids=seeds)
    else:
        subgraph = build_environment_capability_graph_from_wiring(manifest, snapshot)
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
