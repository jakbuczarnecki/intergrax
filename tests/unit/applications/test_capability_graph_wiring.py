# © Artur Czarnecki. All rights reserved.

"""CG-1/2: Environment capability graph wiring and assembly validation."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.capability_graph_assembly_resolver import (
    CapabilityGraphAssemblyError,
    assert_capability_graph_assembly_valid,
    validate_environment_capability_graph,
)
from intergrax.applications._shared.capability_graph_wiring import (
    EnvironmentCapabilityGraphView,
    extract_environment_capability_graph,
    resolve_environment_capability_graph,
)
from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.registry_snapshot import resolve_registry_snapshot
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdge,
    CapabilityEdgeType,
    CapabilityGraph,
    CapabilityNode,
    CapabilityNodeType,
)
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _minimal_catalog() -> CapabilityGraph:
    return CapabilityGraph(
        nodes=[
            CapabilityNode(node_id="application:lab_application", node_type=CapabilityNodeType.APPLICATION),
            CapabilityNode(node_id="tool:rag.retrieve", node_type=CapabilityNodeType.TOOL),
            CapabilityNode(node_id="skill:harness.tool_smoke", node_type=CapabilityNodeType.SKILL),
            CapabilityNode(node_id="agent:echo", node_type=CapabilityNodeType.AGENT),
            CapabilityNode(node_id="policy:runtime_policy_bundle", node_type=CapabilityNodeType.POLICY),
        ],
        edges=[
            CapabilityEdge(
                source_node_id="skill:harness.tool_smoke",
                target_node_id="tool:rag.retrieve",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            ),
            CapabilityEdge(
                source_node_id="agent:echo",
                target_node_id="skill:harness.tool_smoke",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            ),
            CapabilityEdge(
                source_node_id="application:lab_application",
                target_node_id="agent:echo",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            ),
        ],
    )


def test_extract_environment_capability_graph_includes_neighbors() -> None:
    catalog = _minimal_catalog()
    subgraph = extract_environment_capability_graph(
        catalog,
        seed_node_ids=frozenset({"agent:echo"}),
    )
    node_ids = {node.node_id for node in subgraph.nodes}
    assert "agent:echo" in node_ids
    assert "skill:harness.tool_smoke" in node_ids
    assert "tool:rag.retrieve" in node_ids


def test_resolve_environment_capability_graph_from_lab_wiring() -> None:
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="cg.resolve")
    wiring = wire_application_environment(build_lab_manifest(settings), env)
    view = resolve_environment_capability_graph(
        build_lab_manifest(settings),
        env,
        wiring.registry_snapshot,  # type: ignore[arg-type]
    )
    assert view.contains_node("application:lab_application")
    assert any(node_id.startswith("agent:") for node_id in view.node_ids())


def test_wire_application_environment_includes_capability_graph() -> None:
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="cg.wire")
    wiring = wire_application_environment(build_lab_manifest(settings), env)

    assert wiring.capability_graph is not None
    assert wiring.capability_graph.graph.nodes


def test_validate_environment_capability_graph_detects_missing_tool_node() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="cg.missing-tool")
    wiring = wire_application_environment(manifest, env)
    snapshot = wiring.registry_snapshot
    assert snapshot is not None

    empty_view = EnvironmentCapabilityGraphView(
        graph=CapabilityGraph(
            nodes=[
                CapabilityNode(
                    node_id="application:lab_application",
                    node_type=CapabilityNodeType.APPLICATION,
                ),
            ],
            edges=[],
        ),
    )
    result = validate_environment_capability_graph(empty_view, snapshot, manifest)
    assert not result.valid
    assert any("tool:" in error or "skill:" in error or "agent:" in error for error in result.errors)


def test_assert_capability_graph_assembly_valid_raises() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="cg.raise")
    wiring = wire_application_environment(manifest, env)
    snapshot = resolve_registry_snapshot(wiring.build_context)
    view = EnvironmentCapabilityGraphView(graph=CapabilityGraph(nodes=[], edges=[]))

    with pytest.raises(CapabilityGraphAssemblyError):
        assert_capability_graph_assembly_valid(view, snapshot, manifest)
