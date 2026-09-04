# © Artur Czarnecki. All rights reserved.

"""
Harness CFG simulation — abstract multi-agent pipeline (ORCH-CONFIG).

Maps the session's *business narrative* (evidence review → correspondence draft) to
generic acceptance stubs. No Tier-2/Tier-3 product implementation required.
"""

from __future__ import annotations

import json

import pytest

from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    OrchestrationProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.runtime.architecture.multi_agent_coordination import CoordinationPattern
from intergrax.applications.contracts.graph_spec import (
    ApplicationGraphSpec,
    GraphEdge,
    GraphEdgeKind,
    GraphNode,
)
from intergrax.applications.contracts.intent_route import IntentRoute
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_PIPELINE = "acceptance.harness.pipeline"
_SINGLE = "acceptance.harness.intake"


def _simulation_environment() -> ApplicationEnvironmentProfile:
    """CFG-06 / CFG-04 / CFG-18 profile — product-agnostic harness wiring."""
    return ApplicationEnvironmentProfile.lab_defaults(profile_id="harness.cfg.sim").model_copy(
        update={
            "graph_spec": ApplicationGraphSpec(
                nodes=[
                    GraphNode(agent_id="evidence_agent"),
                    GraphNode(agent_id="response_agent"),
                ],
                edges=[
                    GraphEdge(
                        source_agent_id="evidence_agent",
                        target_agent_id="response_agent",
                        kind=GraphEdgeKind.DEPENDS_ON,
                    ),
                ],
                trigger_capabilities=[_PIPELINE],
            ),
            "orchestration_profile": OrchestrationProfile(
                classifier_kind="rules",
                merge_strategy="structured_json",
                intent_routes=[
                    IntentRoute(
                        capability=_PIPELINE,
                        keywords=["podwykonaw", "pismo", "odpisa", "zapłat", "subcontractor"],
                    ),
                    IntentRoute(
                        capability=_SINGLE,
                        keywords=["index", "ingest", "załącz"],
                    ),
                ],
            ),
        }
    )


def _simulation_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(
        UaepPipelineStubAgent(
            agent_id="evidence_agent",
            capability="evidence.analyze",
            prefix="evidence",
            extra_capabilities=(_PIPELINE,),
            description="harness cfg simulation stub",
        )
    )
    registry.register(
        UaepPipelineStubAgent(
            agent_id="response_agent",
            capability="correspondence.draft",
            prefix="response",
            extra_capabilities=(_PIPELINE,),
            description="harness cfg simulation stub",
        )
    )
    registry.register(
        UaepPipelineStubAgent(
            agent_id="intake_agent",
            capability=_SINGLE,
            prefix="intake",
            extra_capabilities=(_PIPELINE,),
            description="harness cfg simulation stub",
        )
    )
    return registry


def _simulation_loop() -> NexusLoop:
    return build_nexus_loop_from_environment(
        _simulation_registry(),
        env=_simulation_environment(),
    )


@pytest.mark.asyncio
async def test_cfg06_sequential_pipeline_two_agents() -> None:
    """CFG-06: orchestration token → graph_spec seed → sequential execution."""
    loop = _simulation_loop()
    result = await loop.handle_task(
        Task(
            tenant_id="org-sim",
            user_id="operator-1",
            message="review bundled evidence",
            context=TaskContext(capability=_PIPELINE),
            metadata={"case_ref": "sim-001"},
        ),
    )
    assert result.state == TaskState.COMPLETED
    payload = json.loads(result.answer)
    assert len(payload["agents"]) == 2
    assert payload["agents"][0]["agent_id"] == "evidence_agent"
    assert payload["agents"][1]["agent_id"] == "response_agent"
    assert result.metadata.get("agent_ids") == ["evidence_agent", "response_agent"]


@pytest.mark.asyncio
async def test_cfg04_free_text_rules_route_to_pipeline() -> None:
    """CFG-04: session narrative (PL) → rules classifier → pipeline graph."""
    loop = _simulation_loop()
    message = (
        "Mamy podwykonawcę XYZ który domaga się zapłaty za wadliwe prace. "
        "Wysłał pismo z wezwaniem. Jak odpisać?"
    )
    result = await loop.handle_task(
        Task(
            tenant_id="org-sim",
            user_id="operator-1",
            message=message,
            context=TaskContext(),
        ),
    )
    assert result.state == TaskState.COMPLETED
    assert result.metadata.get("agent_ids") == ["evidence_agent", "response_agent"]


def _three_agent_sequential_environment() -> ApplicationEnvironmentProfile:
    """CFG-07: N=3 sequential graph."""
    return _simulation_environment().model_copy(
        update={
            "graph_spec": ApplicationGraphSpec(
                nodes=[
                    GraphNode(agent_id="evidence_agent"),
                    GraphNode(agent_id="response_agent"),
                    GraphNode(agent_id="synthesis_agent"),
                ],
                edges=[
                    GraphEdge(
                        source_agent_id="evidence_agent",
                        target_agent_id="response_agent",
                        kind=GraphEdgeKind.DEPENDS_ON,
                    ),
                    GraphEdge(
                        source_agent_id="response_agent",
                        target_agent_id="synthesis_agent",
                        kind=GraphEdgeKind.DEPENDS_ON,
                    ),
                ],
                trigger_capabilities=[_PIPELINE],
            ),
        }
    )


def _three_agent_parallel_environment() -> ApplicationEnvironmentProfile:
    """CFG-08: N=3 parallel batch (no DEPENDS_ON edges)."""
    return _simulation_environment().model_copy(
        update={
            "graph_spec": ApplicationGraphSpec(
                nodes=[
                    GraphNode(agent_id="evidence_agent"),
                    GraphNode(agent_id="response_agent"),
                    GraphNode(agent_id="synthesis_agent"),
                ],
                edges=[],
                trigger_capabilities=[_PIPELINE],
            ),
            "orchestration_profile": OrchestrationProfile(
                classifier_kind="rules",
                merge_strategy="structured_json",
                max_parallel_nodes=3,
                intent_routes=[
                    IntentRoute(
                        capability=_PIPELINE,
                        keywords=["podwykonaw", "pismo", "odpisa", "zapłat", "subcontractor"],
                    ),
                    IntentRoute(
                        capability=_SINGLE,
                        keywords=["index", "ingest", "załącz"],
                    ),
                ],
            ),
        }
    )


def _three_agent_registry() -> AgentRegistry:
    registry = _simulation_registry()
    registry.register(
        UaepPipelineStubAgent(
            agent_id="synthesis_agent",
            capability="synthesis.merge",
            prefix="synthesis",
            extra_capabilities=(_PIPELINE,),
            description="harness cfg simulation stub",
        )
    )
    return registry


@pytest.mark.asyncio
async def test_cfg07_three_agent_sequential_graph() -> None:
    """CFG-07: graph_spec with three DEPENDS_ON layers."""
    loop = build_nexus_loop_from_environment(
        _three_agent_registry(),
        env=_three_agent_sequential_environment(),
    )
    result = await loop.handle_task(
        Task(
            tenant_id="org-sim",
            user_id="operator-1",
            message="sequential three-agent review",
            context=TaskContext(capability=_PIPELINE),
        ),
    )
    assert result.state == TaskState.COMPLETED
    assert result.metadata.get("agent_ids") == [
        "evidence_agent",
        "response_agent",
        "synthesis_agent",
    ]


@pytest.mark.asyncio
async def test_cfg08_three_agent_parallel_graph() -> None:
    """CFG-08: parallel batch when graph has no DEPENDS_ON edges."""
    loop = build_nexus_loop_from_environment(
        _three_agent_registry(),
        env=_three_agent_parallel_environment(),
    )
    result = await loop.handle_task(
        Task(
            tenant_id="org-sim",
            user_id="operator-1",
            message="parallel three-agent review",
            context=TaskContext(capability=_PIPELINE),
        ),
    )
    assert result.state == TaskState.COMPLETED
    agent_ids = result.metadata.get("agent_ids")
    assert agent_ids is not None
    assert set(agent_ids) == {"evidence_agent", "response_agent", "synthesis_agent"}


def _swarm_parallel_environment() -> ApplicationEnvironmentProfile:
    """CFG-17: swarm coordination with three parallel root agents."""
    base = ApplicationEnvironmentProfile.swarm_exploration_defaults(max_parallel_nodes=3)
    intent_routes = [
        IntentRoute(
            capability=_PIPELINE,
            keywords=["podwykonaw", "pismo", "odpisa", "zapłat", "subcontractor"],
        ),
        IntentRoute(
            capability=_SINGLE,
            keywords=["index", "ingest", "załącz"],
        ),
    ]
    return base.model_copy(
        update={
            "graph_spec": ApplicationGraphSpec(
                nodes=[
                    GraphNode(agent_id="evidence_agent"),
                    GraphNode(agent_id="response_agent"),
                    GraphNode(agent_id="synthesis_agent"),
                ],
                edges=[],
                trigger_capabilities=[_PIPELINE],
            ),
            "orchestration_profile": base.orchestration_profile.model_copy(
                update={
                    "classifier_kind": "rules",
                    "intent_routes": intent_routes,
                },
            ),
        },
    )


def _strict_multi_agent_environment() -> ApplicationEnvironmentProfile:
    """CFG-20: strict preset with two-agent sequential graph."""
    base = ApplicationEnvironmentProfile.strict_multi_agent_defaults(
        profile_id="harness.cfg.strict",
    )
    return base.model_copy(
        update={
            "graph_spec": ApplicationGraphSpec(
                nodes=[
                    GraphNode(agent_id="evidence_agent"),
                    GraphNode(agent_id="response_agent"),
                ],
                edges=[
                    GraphEdge(
                        source_agent_id="evidence_agent",
                        target_agent_id="response_agent",
                        kind=GraphEdgeKind.DEPENDS_ON,
                    ),
                ],
                trigger_capabilities=[_PIPELINE],
            ),
            "orchestration_profile": base.orchestration_profile.model_copy(
                update={
                    "classifier_kind": "rules",
                    "intent_routes": [
                        IntentRoute(
                            capability=_PIPELINE,
                            keywords=["podwykonaw", "pismo", "odpisa", "zapłat", "subcontractor"],
                        ),
                    ],
                },
            ),
        },
    )


@pytest.mark.asyncio
async def test_cfg17_swarm_parallel_graph() -> None:
    """CFG-17: swarm-labelled profile executes three-agent parallel batch."""
    loop = build_nexus_loop_from_environment(
        _three_agent_registry(),
        env=_swarm_parallel_environment(),
    )
    result = await loop.handle_task(
        Task(
            tenant_id="org-sim",
            user_id="operator-1",
            message="parallel swarm review",
            context=TaskContext(capability=_PIPELINE),
        ),
    )
    assert result.state == TaskState.COMPLETED
    assert result.metadata.get("coordination_pattern") == CoordinationPattern.SWARM.value
    agent_ids = result.metadata.get("agent_ids")
    assert agent_ids is not None
    assert set(agent_ids) == {"evidence_agent", "response_agent", "synthesis_agent"}


@pytest.mark.asyncio
async def test_cfg20_strict_multi_agent_pipeline() -> None:
    """CFG-20: strict_multi_agent_defaults completes structured two-agent graph."""
    env = _strict_multi_agent_environment()
    assert env.execution_mode is ExecutionMode.STRICT
    assert env.decision_profile.verification.semantic_enabled is True
    assert env.decision_profile.flow.verify_graph_final is True
    loop = build_nexus_loop_from_environment(_simulation_registry(), env=env)
    result = await loop.handle_task(
        Task(
            tenant_id="org-sim",
            user_id="operator-1",
            message="strict multi-agent review",
            context=TaskContext(capability=_PIPELINE),
        ),
    )
    assert result.state == TaskState.COMPLETED
    payload = json.loads(result.answer)
    assert len(payload["agents"]) == 2


@pytest.mark.asyncio
async def test_cfg18_single_route_not_replaced_by_graph() -> None:
    """CFG-18: intake capability must not trigger graph_spec seed."""
    loop = _simulation_loop()
    result = await loop.handle_task(
        Task(
            tenant_id="org-sim",
            user_id="operator-1",
            message="załącz dokumenty sprawy",
            context=TaskContext(capability=_SINGLE),
        ),
    )
    assert result.state == TaskState.COMPLETED
    assert result.metadata.get("agent_ids") == ["intake_agent"]
