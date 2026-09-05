# © Artur Czarnecki. All rights reserved.

"""APP-OPS-1 — STRICT capability graph deploy gate."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.capability_graph_deploy_gate import (
    build_environment_capability_deploy_report,
    check_strict_product_capability_graph,
    validate_strict_capability_graph_deploy,
)
from intergrax.applications._shared.capability_graph_wiring import EnvironmentCapabilityGraphView
from intergrax.applications._shared.product_manifest_registry import iter_strict_product_manifests
from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdge,
    CapabilityEdgeType,
    CapabilityGraph,
    CapabilityNode,
    CapabilityNodeType,
)
from echo.echo_agent import EchoAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def _graph_view() -> EnvironmentCapabilityGraphView:
    return EnvironmentCapabilityGraphView(
        graph=CapabilityGraph(
            nodes=[
                CapabilityNode(
                    node_id="application:gate_test_application",
                    node_type=CapabilityNodeType.APPLICATION,
                ),
                CapabilityNode(node_id="agent:echo", node_type=CapabilityNodeType.AGENT),
                CapabilityNode(node_id="tool:rag.retrieve", node_type=CapabilityNodeType.TOOL),
            ],
            edges=[
                CapabilityEdge(
                    source_node_id="application:gate_test_application",
                    target_node_id="agent:echo",
                    edge_type=CapabilityEdgeType.DEPENDS_ON,
                ),
                CapabilityEdge(
                    source_node_id="agent:echo",
                    target_node_id="tool:rag.retrieve",
                    edge_type=CapabilityEdgeType.DEPENDS_ON,
                ),
            ],
        ),
    )


def test_build_environment_capability_deploy_report_includes_impact() -> None:
    report = build_environment_capability_deploy_report(_graph_view())
    assert report.lineage.records
    assert any(record.node_id == "agent:echo" for record in report.impact.impacts)


def test_validate_strict_capability_graph_deploy_blocks_experimental_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = ApplicationManifest.lab(
        app_id="gate_test",
        name="Gate Test",
        route_prefix="/v1/gate_test",
        env_prefix="GATE_TEST_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    ).model_copy(update={"app_id": "gate_test"})
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="gate_test.product")
    view = _graph_view()
    snapshot = HarnessRegistrySnapshot(
        integration_profile=env.integration_profile,
        tool_registry=None,
        skill_registry=None,
        prompt_registry=None,
        policy_bundle=None,
        agent_registry=None,
    )
    original_get_contract = EchoAgent.get_contract

    def experimental_contract(self: EchoAgent):
        return original_get_contract(self).model_copy(
            update={"lifecycle_state": AgentLifecycleState.EXPERIMENTAL}
        )

    monkeypatch.setattr(EchoAgent, "get_contract", experimental_contract)

    result = validate_strict_capability_graph_deploy(view, snapshot, manifest, env)
    assert not result.valid
    assert any("STRICT deploy blocks roster agent" in error for error in result.errors)


@pytest.mark.parametrize(
    ("product_id", "manifest"),
    list(iter_strict_product_manifests()),
    ids=[product_id for product_id, _ in iter_strict_product_manifests()],
)
def test_strict_product_capability_graph_deploy_gate(product_id: str, manifest) -> None:
    violations = check_strict_product_capability_graph(product_id, manifest)
    assert violations == [], "\n".join(violations)


def test_check_strict_product_capability_graph_skips_non_strict_manifests() -> None:
    manifest = ApplicationManifest.lab(
        app_id="lab_skip_gate",
        name="Lab Skip Gate",
        route_prefix="/v1/lab_skip_gate",
        env_prefix="LAB_SKIP_GATE_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    assert check_strict_product_capability_graph("lab_skip_gate", manifest) == []
