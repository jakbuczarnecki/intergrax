# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from intergrax.agent_distribution.agent_capability_metadata import (
    AgentCapabilityDescriptor,
)
from intergrax.runtime.architecture.capability_graph import (
    CapabilityNodeType,
    build_catalog_capability_graph,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CAPABILITY_GRAPH_MODULE = REPO_ROOT / "intergrax" / "runtime" / "architecture" / "capability_graph.py"

_FORBIDDEN_FRAGMENTS = (
    "AgentRegistry",
    "testing_support.agent_registry_bootstrap",
    "echo.echo_agent",
    "EchoAgent",
    "research.research_agent",
    "legal.legal_agent",
    "importlib.import_module",
    "contract.version if contract is not None else",
    "default_agent_capability_metadata_provider",
    "_BUILTIN_AGENT_CAPABILITY_DESCRIPTORS",
)


class _FakeAgentCapabilityMetadataProvider:
    def __init__(self, descriptors: Sequence[AgentCapabilityDescriptor]) -> None:
        self._descriptors = tuple(descriptors)

    def list_agent_capability_descriptors(self) -> Sequence[AgentCapabilityDescriptor]:
        return self._descriptors


def test_external_agents_discovered_via_metadata_provider_without_core_edits() -> None:
    provider = _FakeAgentCapabilityMetadataProvider(
        (
            AgentCapabilityDescriptor(
                contract_id="agent-a",
                agent_version="2.3.1",
                capabilities=("knowledge.search",),
                skill_ids=("harness.tool_smoke",),
                tool_ids=("rag.retrieve",),
            ),
            AgentCapabilityDescriptor(
                contract_id="agent-b",
                agent_version="7.0.4",
            ),
        ),
    )
    graph = build_catalog_capability_graph(agent_metadata_provider=provider)

    agent_nodes = {
        node.node_id: node
        for node in graph.nodes
        if node.node_type == CapabilityNodeType.AGENT
    }
    assert "agent:agent-a" in agent_nodes
    assert "agent:agent-b" in agent_nodes
    assert agent_nodes["agent:agent-a"].version == "2.3.1"
    assert agent_nodes["agent:agent-b"].version == "7.0.4"
    assert agent_nodes["agent:agent-a"].metadata["capabilities"] == "knowledge.search"

    edge_keys = {(edge.source_node_id, edge.target_node_id) for edge in graph.edges}
    assert ("agent:agent-a", "skill:harness.tool_smoke") in edge_keys
    assert ("agent:agent-a", "tool:rag.retrieve") in edge_keys


def test_capability_graph_module_has_no_runtime_agent_discovery_debt() -> None:
    text = CAPABILITY_GRAPH_MODULE.read_text(encoding="utf-8")
    violations = [
        f"contains forbidden fragment {fragment!r}"
        for fragment in _FORBIDDEN_FRAGMENTS
        if fragment in text
    ]
    assert not violations, "\n".join(violations)


def test_capability_graph_module_does_not_seed_agents_from_harness_catalog() -> None:
    text = CAPABILITY_GRAPH_MODULE.read_text(encoding="utf-8")
    assert "HARNESS_CAPABILITY_CATALOG" not in text
    assert "harness_capability_catalog" not in text
