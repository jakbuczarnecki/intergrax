# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from intergrax.agent_distribution.agent_capability_metadata import AgentCapabilityDescriptor
from intergrax.agent_distribution.builtin_capability_metadata import (
    PackageAgentCapabilityMetadataProvider,
)
from intergrax.contracts.application_capability_metadata import (
    ApplicationCapabilityDescriptor,
    ApplicationCapabilityProjectionConflict,
)
from intergrax.applications.reference.builtin_application_capability_metadata import (
    HarnessReferenceApplicationCapabilityMetadataProvider,
)
from intergrax.applications.reference.harness_manifest_catalog import build_harness_reference_manifests
from intergrax.runtime.architecture.capability_graph import (
    CapabilityNodeType,
    build_catalog_capability_graph,
)
from intergrax.runtime.architecture.capability_graph_applications import application_capability_node_id
from intergrax.runtime.architecture.capability_graph_query import CapabilityGraphQuery

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PACKAGE_METADATA_PROVIDER = PackageAgentCapabilityMetadataProvider(
    package_roots=(
        _REPO_ROOT / "agents" / "echo",
        _REPO_ROOT / "agents" / "legal",
        _REPO_ROOT / "agents" / "research",
    )
)
_APPLICATION_METADATA_PROVIDER = HarnessReferenceApplicationCapabilityMetadataProvider()


class _FakeApplicationCapabilityMetadataProvider:
    def __init__(self, descriptors: Sequence[ApplicationCapabilityDescriptor]) -> None:
        self._descriptors = tuple(descriptors)

    def list_application_capability_descriptors(self) -> Sequence[ApplicationCapabilityDescriptor]:
        return self._descriptors


class _FakeAgentCapabilityMetadataProvider:
    def __init__(self, descriptors: Sequence[AgentCapabilityDescriptor]) -> None:
        self._descriptors = tuple(descriptors)

    def list_agent_capability_descriptors(self) -> Sequence[AgentCapabilityDescriptor]:
        return self._descriptors


def test_no_application_provider_yields_no_application_inventory_nodes() -> None:
    graph = build_catalog_capability_graph(agent_metadata_provider=_PACKAGE_METADATA_PROVIDER)
    application_nodes = [
        node for node in graph.nodes if node.node_type == CapabilityNodeType.APPLICATION
    ]
    assert application_nodes == []


def test_injected_application_provider_yields_application_nodes() -> None:
    graph = build_catalog_capability_graph(
        agent_metadata_provider=_PACKAGE_METADATA_PROVIDER,
        application_metadata_provider=_APPLICATION_METADATA_PROVIDER,
    )
    application_node_ids = {
        node.node_id
        for node in graph.nodes
        if node.node_type == CapabilityNodeType.APPLICATION
    }
    assert application_capability_node_id("lab") in application_node_ids
    assert application_capability_node_id("legal") in application_node_ids
    assert application_capability_node_id("research") in application_node_ids
    assert application_capability_node_id("poc_template") in application_node_ids


def test_application_agent_edges_are_scoped_per_manifest() -> None:
    graph = build_catalog_capability_graph(
        agent_metadata_provider=_PACKAGE_METADATA_PROVIDER,
        application_metadata_provider=_APPLICATION_METADATA_PROVIDER,
    )
    edge_keys = {
        (edge.source_node_id, edge.target_node_id)
        for edge in graph.edges
        if edge.source_node_id.startswith("application:")
    }

    legal_node = application_capability_node_id("legal")
    lab_node = application_capability_node_id("lab")

    assert (legal_node, "agent:legal") in edge_keys
    assert (lab_node, "agent:echo") in edge_keys
    assert (legal_node, "agent:echo") not in edge_keys
    assert (lab_node, "agent:legal") not in edge_keys


def test_research_agents_link_to_research_application_not_lab() -> None:
    graph = build_catalog_capability_graph(
        agent_metadata_provider=_PACKAGE_METADATA_PROVIDER,
        application_metadata_provider=_APPLICATION_METADATA_PROVIDER,
    )
    edge_keys = {
        (edge.source_node_id, edge.target_node_id)
        for edge in graph.edges
        if edge.source_node_id.startswith("application:")
    }
    research_node = application_capability_node_id("research")
    lab_node = application_capability_node_id("lab")

    assert (research_node, "agent:research") in edge_keys
    assert (research_node, "agent:research-summary") in edge_keys
    assert (lab_node, "agent:research") not in edge_keys


def test_missing_referenced_agent_metadata_fails_closed() -> None:
    provider = _FakeApplicationCapabilityMetadataProvider(
        (
            ApplicationCapabilityDescriptor(
                application_id="lab",
                application_version="1.0.0",
                agent_contract_ids=("missing-agent",),
            ),
        ),
    )
    with pytest.raises(ApplicationCapabilityProjectionConflict, match="missing-agent"):
        build_catalog_capability_graph(
            agent_metadata_provider=_FakeAgentCapabilityMetadataProvider(()),
            application_metadata_provider=provider,
        )


def test_no_hardcoded_lab_legal_research_poc_application_nodes_without_provider() -> None:
    graph = build_catalog_capability_graph()
    hardcoded = {
        application_capability_node_id(app_id)
        for app_id in ("lab", "legal", "research", "poc_template")
    }
    node_ids = {node.node_id for node in graph.nodes}
    assert hardcoded.isdisjoint(node_ids)


def test_product_edges_generated_from_injected_application_descriptors() -> None:
    graph = build_catalog_capability_graph(
        agent_metadata_provider=_PACKAGE_METADATA_PROVIDER,
        application_metadata_provider=_APPLICATION_METADATA_PROVIDER,
    )
    product_edges = {
        edge.target_node_id
        for edge in graph.edges
        if edge.source_node_id == "product:intergrax_harness"
    }
    assert application_capability_node_id("lab") in product_edges
    assert application_capability_node_id("legal") in product_edges


def test_capability_graph_query_agents_for_application() -> None:
    graph = build_catalog_capability_graph(
        agent_metadata_provider=_PACKAGE_METADATA_PROVIDER,
        application_metadata_provider=_APPLICATION_METADATA_PROVIDER,
    )
    query = CapabilityGraphQuery(graph)
    assert query.agents_for_application("research") == ("research", "research-summary")


def test_capability_graph_query_applications_for_agent() -> None:
    graph = build_catalog_capability_graph(
        agent_metadata_provider=_PACKAGE_METADATA_PROVIDER,
        application_metadata_provider=_APPLICATION_METADATA_PROVIDER,
    )
    query = CapabilityGraphQuery(graph)
    assert query.applications_for_agent("echo") == ("lab", "poc_template")


def test_capability_graph_query_capabilities_for_agent() -> None:
    graph = build_catalog_capability_graph(
        agent_metadata_provider=_PACKAGE_METADATA_PROVIDER,
        application_metadata_provider=_APPLICATION_METADATA_PROVIDER,
    )
    query = CapabilityGraphQuery(graph)
    assert query.capabilities_for_agent("echo") == ("echo.basic",)


def test_capability_graph_query_agents_with_capability() -> None:
    provider = _FakeAgentCapabilityMetadataProvider(
        (
            AgentCapabilityDescriptor(
                contract_id="agent-a",
                agent_version="1.0.0",
                capabilities=("knowledge.search",),
            ),
            AgentCapabilityDescriptor(
                contract_id="agent-b",
                agent_version="1.0.0",
                capabilities=("other.cap",),
            ),
        ),
    )
    graph = build_catalog_capability_graph(agent_metadata_provider=provider)
    query = CapabilityGraphQuery(graph)
    assert query.agents_with_capability("knowledge.search") == ("agent-a",)


def test_capability_graph_query_deterministic_ordering() -> None:
    graph = build_catalog_capability_graph(
        agent_metadata_provider=_PACKAGE_METADATA_PROVIDER,
        application_metadata_provider=_APPLICATION_METADATA_PROVIDER,
    )
    query = CapabilityGraphQuery(graph)
    first = query.agents_for_application("research")
    second = query.agents_for_application("research")
    assert first == second == ("research", "research-summary")


def test_harness_reference_manifest_count_matches_provider() -> None:
    descriptors = _APPLICATION_METADATA_PROVIDER.list_application_capability_descriptors()
    assert len(descriptors) == len(build_harness_reference_manifests())
