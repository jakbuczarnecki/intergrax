# © Artur Czarnecki. All rights reserved.

"""Capability graph contracts and baseline catalog graph builder (Phase V-CG.1)."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Iterable

from pydantic import BaseModel, Field, model_validator

from intergrax.agent_distribution.agent_capability_metadata import (
    AgentCapabilityDescriptor,
    AgentCapabilityMetadataProvider,
    merge_agent_capability_descriptors,
)
from intergrax.contracts.application_capability_metadata import (
    ApplicationCapabilityDescriptor,
    ApplicationCapabilityMetadataProvider,
    merge_application_capability_descriptors,
)
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.catalog import list_slugs
from intergrax.skills.registry.bootstrap import register_default_skills
from intergrax.skills.registry.factory import build_registry_from_profile
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.registry.bootstrap import register_default_tools
from intergrax.tools.registry.catalog import list_catalog_tool_ids


class CapabilityNodeType(str, Enum):
    INTEGRATION = "integration"
    TOOL = "tool"
    SKILL = "skill"
    POLICY = "policy"
    AGENT = "agent"
    APPLICATION = "application"
    PRODUCT = "product"
    PROMPT = "prompt"
    EVALUATION = "evaluation"


class CapabilityEdgeType(str, Enum):
    DEPENDS_ON = "depends_on"
    CONSTRAINED_BY = "constrained_by"
    EVALUATES = "evaluates"
    SUPERSEDES = "supersedes"
    COMPATIBLE_WITH = "compatible_with"


class CapabilityGraphVersion(BaseModel):
    schema_version: str = "1.0.0"
    graph_version: str = "1.0.0"


class CapabilityNode(BaseModel):
    node_id: str
    node_type: CapabilityNodeType
    version: str = "1.0.0"
    metadata: dict[str, str] = Field(default_factory=dict)


class CapabilityEdge(BaseModel):
    source_node_id: str
    target_node_id: str
    edge_type: CapabilityEdgeType
    metadata: dict[str, str] = Field(default_factory=dict)


class CapabilityGraph(BaseModel):
    version: CapabilityGraphVersion = Field(default_factory=CapabilityGraphVersion)
    nodes: list[CapabilityNode] = Field(default_factory=list)
    edges: list[CapabilityEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> "CapabilityGraph":
        node_ids = [node.node_id for node in self.nodes]
        unique_node_ids = set(node_ids)
        if len(unique_node_ids) != len(node_ids):
            raise ValueError("CapabilityGraph contains duplicate node_id values")

        node_types = {node.node_id: node.node_type for node in self.nodes}
        for edge in self.edges:
            if edge.source_node_id not in unique_node_ids:
                raise ValueError(f"Edge source is not present in graph: {edge.source_node_id}")
            if edge.target_node_id not in unique_node_ids:
                raise ValueError(f"Edge target is not present in graph: {edge.target_node_id}")
            if not _is_allowed_edge(
                source=node_types[edge.source_node_id],
                edge_type=edge.edge_type,
                target=node_types[edge.target_node_id],
            ):
                raise ValueError(
                    "Invalid edge relation: "
                    f"{edge.source_node_id} ({node_types[edge.source_node_id].value}) "
                    f"-[{edge.edge_type.value}]-> "
                    f"{edge.target_node_id} ({node_types[edge.target_node_id].value})"
                )
        return self


_ALLOWED_EDGES: dict[CapabilityEdgeType, set[tuple[CapabilityNodeType, CapabilityNodeType]]] = {
    CapabilityEdgeType.DEPENDS_ON: {
        (CapabilityNodeType.TOOL, CapabilityNodeType.INTEGRATION),
        (CapabilityNodeType.SKILL, CapabilityNodeType.TOOL),
        (CapabilityNodeType.AGENT, CapabilityNodeType.SKILL),
        (CapabilityNodeType.AGENT, CapabilityNodeType.TOOL),
        (CapabilityNodeType.APPLICATION, CapabilityNodeType.AGENT),
        (CapabilityNodeType.PRODUCT, CapabilityNodeType.APPLICATION),
        (CapabilityNodeType.PROMPT, CapabilityNodeType.SKILL),
    },
    CapabilityEdgeType.CONSTRAINED_BY: {
        (CapabilityNodeType.TOOL, CapabilityNodeType.POLICY),
        (CapabilityNodeType.SKILL, CapabilityNodeType.POLICY),
        (CapabilityNodeType.AGENT, CapabilityNodeType.POLICY),
        (CapabilityNodeType.APPLICATION, CapabilityNodeType.POLICY),
    },
    CapabilityEdgeType.EVALUATES: {
        (CapabilityNodeType.EVALUATION, CapabilityNodeType.TOOL),
        (CapabilityNodeType.EVALUATION, CapabilityNodeType.SKILL),
        (CapabilityNodeType.EVALUATION, CapabilityNodeType.AGENT),
        (CapabilityNodeType.EVALUATION, CapabilityNodeType.APPLICATION),
    },
    CapabilityEdgeType.SUPERSEDES: {
        (CapabilityNodeType.TOOL, CapabilityNodeType.TOOL),
        (CapabilityNodeType.SKILL, CapabilityNodeType.SKILL),
        (CapabilityNodeType.AGENT, CapabilityNodeType.AGENT),
        (CapabilityNodeType.POLICY, CapabilityNodeType.POLICY),
    },
    CapabilityEdgeType.COMPATIBLE_WITH: {
        (CapabilityNodeType.TOOL, CapabilityNodeType.TOOL),
        (CapabilityNodeType.SKILL, CapabilityNodeType.SKILL),
        (CapabilityNodeType.AGENT, CapabilityNodeType.AGENT),
        (CapabilityNodeType.APPLICATION, CapabilityNodeType.APPLICATION),
    },
}


def _is_allowed_edge(
    *,
    source: CapabilityNodeType,
    edge_type: CapabilityEdgeType,
    target: CapabilityNodeType,
) -> bool:
    return (source, target) in _ALLOWED_EDGES[edge_type]


def _integration_nodes() -> list[CapabilityNode]:
    register_default_integrations()
    return [
        CapabilityNode(
            node_id=f"integration:{slug}",
            node_type=CapabilityNodeType.INTEGRATION,
        )
        for slug in list_slugs()
    ]


def _tool_nodes() -> list[CapabilityNode]:
    register_default_tools()
    return [
        CapabilityNode(
            node_id=f"tool:{tool_id}",
            node_type=CapabilityNodeType.TOOL,
        )
        for tool_id in list_catalog_tool_ids()
    ]


def _skill_nodes_and_edges() -> tuple[list[CapabilityNode], list[CapabilityEdge]]:
    register_default_skills()
    skill_registry = build_registry_from_profile(
        SkillProfile(register_all_catalog_bundles=True),
    )
    nodes: list[CapabilityNode] = []
    edges: list[CapabilityEdge] = []
    prompt_ids: set[str] = set()
    policy_fragment_ids: set[str] = set()

    for entry in skill_registry.list():
        manifest = entry.manifest
        skill_node_id = f"skill:{manifest.skill_id}"
        nodes.append(
            CapabilityNode(
                node_id=skill_node_id,
                node_type=CapabilityNodeType.SKILL,
                version=manifest.version,
            )
        )
        for tool_id in manifest.tool_ids:
            edges.append(
                CapabilityEdge(
                    source_node_id=skill_node_id,
                    target_node_id=f"tool:{tool_id}",
                    edge_type=CapabilityEdgeType.DEPENDS_ON,
                )
            )
        for prompt_id in manifest.prompt_instruction_ids:
            prompt_ids.add(prompt_id)
            edges.append(
                CapabilityEdge(
                    source_node_id=f"prompt:{prompt_id}",
                    target_node_id=skill_node_id,
                    edge_type=CapabilityEdgeType.DEPENDS_ON,
                )
            )
        if manifest.policy_fragment_id:
            policy_fragment_ids.add(manifest.policy_fragment_id)
            edges.append(
                CapabilityEdge(
                    source_node_id=skill_node_id,
                    target_node_id=f"policy:{manifest.policy_fragment_id}",
                    edge_type=CapabilityEdgeType.CONSTRAINED_BY,
                )
            )

    for prompt_id in sorted(prompt_ids):
        nodes.append(
            CapabilityNode(
                node_id=f"prompt:{prompt_id}",
                node_type=CapabilityNodeType.PROMPT,
            )
        )
    for policy_id in sorted(policy_fragment_ids):
        nodes.append(
            CapabilityNode(
                node_id=f"policy:{policy_id}",
                node_type=CapabilityNodeType.POLICY,
            )
        )
    return nodes, edges


def _agent_nodes_and_edges(
    descriptors: Sequence[AgentCapabilityDescriptor],
) -> tuple[list[CapabilityNode], list[CapabilityEdge]]:
    nodes: list[CapabilityNode] = []
    edges: list[CapabilityEdge] = []
    for descriptor in descriptors:
        agent_node_id = f"agent:{descriptor.contract_id}"
        metadata: dict[str, str] = {}
        if descriptor.capabilities:
            metadata["capabilities"] = ",".join(descriptor.capabilities)
        nodes.append(
            CapabilityNode(
                node_id=agent_node_id,
                node_type=CapabilityNodeType.AGENT,
                version=descriptor.agent_version,
                metadata=metadata,
            )
        )
        for skill_id in descriptor.skill_ids:
            edges.append(
                CapabilityEdge(
                    source_node_id=agent_node_id,
                    target_node_id=f"skill:{skill_id}",
                    edge_type=CapabilityEdgeType.DEPENDS_ON,
                )
            )
        for tool_id in descriptor.tool_ids:
            edges.append(
                CapabilityEdge(
                    source_node_id=agent_node_id,
                    target_node_id=f"tool:{tool_id}",
                    edge_type=CapabilityEdgeType.DEPENDS_ON,
                )
            )
    return nodes, edges


def _platform_static_nodes() -> list[CapabilityNode]:
    """Platform concepts that are not derived from injected inventory providers."""
    return [
        CapabilityNode(node_id="policy:runtime_policy_bundle", node_type=CapabilityNodeType.POLICY),
        CapabilityNode(node_id="evaluation:runtime_quality", node_type=CapabilityNodeType.EVALUATION),
        CapabilityNode(node_id="product:intergrax_harness", node_type=CapabilityNodeType.PRODUCT),
    ]


def _modality_compatibility_edges() -> list[CapabilityEdge]:
    """Cross-plane modality tool pairs declared compatible for harness skills."""
    pairs = (
        ("tool:vision.detect", "tool:rag.retrieve"),
        ("tool:vision.detect", "tool:ml.predict"),
        ("tool:ml.predict", "tool:ml.batch_predict"),
        ("tool:speech.synthesize", "tool:speech.transcribe"),
        ("tool:storage.get", "tool:knowledge.search"),
        ("tool:workspace.write_file", "tool:memory.write"),
        ("tool:issues.get_issue", "tool:issues.add_comment"),
        ("tool:issues.create_issue", "tool:issues.get_issue"),
    )
    edges: list[CapabilityEdge] = []
    for left, right in pairs:
        edges.append(
            CapabilityEdge(
                source_node_id=left,
                target_node_id=right,
                edge_type=CapabilityEdgeType.COMPATIBLE_WITH,
            )
        )
        edges.append(
            CapabilityEdge(
                source_node_id=right,
                target_node_id=left,
                edge_type=CapabilityEdgeType.COMPATIBLE_WITH,
            )
        )
    return edges


def _platform_and_inventory_edges(
    *,
    agent_nodes: Sequence[CapabilityNode],
    application_descriptors: Sequence[ApplicationCapabilityDescriptor],
) -> list[CapabilityEdge]:
    from intergrax.runtime.architecture.capability_graph_applications import (
        application_capability_node_id,
        build_application_agent_edges,
    )

    agent_node_ids = frozenset(node.node_id for node in agent_nodes)
    edges: list[CapabilityEdge] = []
    for descriptor in application_descriptors:
        application_node = application_capability_node_id(descriptor.application_id)
        edges.append(
            CapabilityEdge(
                source_node_id=application_node,
                target_node_id="policy:runtime_policy_bundle",
                edge_type=CapabilityEdgeType.CONSTRAINED_BY,
            )
        )
        edges.append(
            CapabilityEdge(
                source_node_id="product:intergrax_harness",
                target_node_id=application_node,
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            )
        )
    edges.extend(
        build_application_agent_edges(
            agent_node_ids=agent_node_ids,
            descriptors=application_descriptors,
        ),
    )
    for node in agent_nodes:
        edges.append(
            CapabilityEdge(
                source_node_id="evaluation:runtime_quality",
                target_node_id=node.node_id,
                edge_type=CapabilityEdgeType.EVALUATES,
            )
        )
    return edges


def _merge_nodes(groups: Iterable[Sequence[CapabilityNode]]) -> list[CapabilityNode]:
    node_by_id: dict[str, CapabilityNode] = {}
    for group in groups:
        for node in group:
            node_by_id[node.node_id] = node
    return sorted(node_by_id.values(), key=lambda item: item.node_id)


def build_catalog_capability_graph(
    *,
    agent_metadata_provider: AgentCapabilityMetadataProvider | None = None,
    application_metadata_provider: ApplicationCapabilityMetadataProvider | None = None,
) -> CapabilityGraph:
    """Build a typed baseline capability graph from current catalogs.

    Agent nodes and agent→skill/tool edges are projected from ``agent_metadata_provider``
    (non-executable declared package metadata). Application nodes and application→agent
    edges are projected from ``application_metadata_provider`` (manifest-derived metadata).
    When a provider is omitted, no inventory is assumed for that plane — there is no
    platform hardcoded agent or application list and no default discovery root.
    """
    if agent_metadata_provider is None:
        agent_descriptors: tuple[AgentCapabilityDescriptor, ...] = ()
    else:
        agent_descriptors = merge_agent_capability_descriptors(
            agent_metadata_provider.list_agent_capability_descriptors(),
        )
    if application_metadata_provider is None:
        application_descriptors: tuple[ApplicationCapabilityDescriptor, ...] = ()
    else:
        application_descriptors = merge_application_capability_descriptors(
            application_metadata_provider.list_application_capability_descriptors(),
        )
    integration_nodes = _integration_nodes()
    tool_nodes = _tool_nodes()
    skill_nodes, skill_edges = _skill_nodes_and_edges()
    agent_nodes, agent_edges = _agent_nodes_and_edges(agent_descriptors)
    from intergrax.runtime.architecture.capability_graph_applications import (
        application_nodes_from_descriptors,
    )

    application_nodes = application_nodes_from_descriptors(application_descriptors)
    platform_nodes = _platform_static_nodes()

    nodes = _merge_nodes(
        [integration_nodes, tool_nodes, skill_nodes, agent_nodes, application_nodes, platform_nodes]
    )
    edges = [
        *skill_edges,
        *agent_edges,
        *_modality_compatibility_edges(),
        *_platform_and_inventory_edges(
            agent_nodes=agent_nodes,
            application_descriptors=application_descriptors,
        ),
    ]
    return CapabilityGraph(nodes=nodes, edges=edges)
