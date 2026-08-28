# © Artur Czarnecki. All rights reserved.

"""Orchestration profile wiring (Phase ORCH-1)."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.orchestration_wiring import (
    GraphSpecSeedingPlanner,
    OrchestrationWiringContext,
    OrchestrationWiringError,
    resolve_max_parallel_nodes,
    resolve_nexus_task_classifier,
    resolve_nexus_task_planner,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    OrchestrationProfile,
)
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec, GraphNode
from intergrax.applications.contracts.intent_route import IntentRoute
from intergrax.runtime.nexus.planning.task_planner import TaskPlanner
from intergrax.runtime.nexus.llm_task_classifier import LlmTaskClassifier
from intergrax.runtime.nexus.rules_task_classifier import RulesTaskClassifier
from intergrax.runtime.nexus.task_classifier import ClassifyingTaskClassifier
from intergrax.runtime.registry.agent_registry import AgentRegistry
from echo.echo_agent import EchoAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _StubLLMAdapter:
    async def generate(self, *args: object, **kwargs: object) -> str:
        return "ok"


def test_resolve_default_planner_and_classifier() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    env = ApplicationEnvironmentProfile.lab_defaults()
    planner = resolve_nexus_task_planner(env)
    classifier = resolve_nexus_task_classifier(registry, env)
    assert isinstance(planner, TaskPlanner)
    assert isinstance(classifier, ClassifyingTaskClassifier)


def test_engine_planner_requires_llm_adapter() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "orchestration_profile": OrchestrationProfile(planner_kind="engine"),
        }
    )
    with pytest.raises(OrchestrationWiringError, match="llm_adapter"):
        resolve_nexus_task_planner(env)


def test_engine_planner_resolves_with_llm_adapter() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "orchestration_profile": OrchestrationProfile(planner_kind="engine"),
        }
    )
    planner = resolve_nexus_task_planner(
        env,
        wiring_context=OrchestrationWiringContext(llm_adapter=_StubLLMAdapter()),  # type: ignore[arg-type]
    )
    assert planner is not None


def test_unknown_planner_kind_fails_fast() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "orchestration_profile": OrchestrationProfile(planner_kind="unknown"),
        }
    )
    with pytest.raises(OrchestrationWiringError, match="Unknown planner_kind"):
        resolve_nexus_task_planner(env)


def test_graph_spec_wraps_planner() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "graph_spec": ApplicationGraphSpec(nodes=[GraphNode(agent_id="EchoAgent")]),
        }
    )
    planner = resolve_nexus_task_planner(env)
    assert isinstance(planner, GraphSpecSeedingPlanner)


def test_graph_spec_planner_reads_live_environment_graph_spec() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "graph_spec": ApplicationGraphSpec(
                nodes=[GraphNode(agent_id="EchoAgent")],
                trigger_capabilities=["echo.basic"],
                evaluator_loop=None,
            ),
        }
    )
    planner = resolve_nexus_task_planner(env)
    assert isinstance(planner, GraphSpecSeedingPlanner)
    from intergrax.applications.contracts.graph_spec import EvaluatorLoopGraphBinding
    from intergrax.runtime.critic.evaluator_loop_spec import EvaluatorLoopSpec

    env.graph_spec = ApplicationGraphSpec(
        nodes=[GraphNode(agent_id="EchoAgent")],
        trigger_capabilities=["echo.basic"],
        evaluator_loop=EvaluatorLoopGraphBinding(
            producer_agent_id="EchoAgent",
            evaluator_agent_id="EchoAgent",
            revise_agent_id="EchoAgent",
            spec=EvaluatorLoopSpec(max_iterations=4, revise_node_id="node_EchoAgent"),
        ),
    )
    assert env.graph_spec.evaluator_loop is not None
    assert env.graph_spec.evaluator_loop.spec.max_iterations == 4


def test_max_parallel_nodes_from_profile() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "orchestration_profile": OrchestrationProfile(max_parallel_nodes=2),
        }
    )
    assert resolve_max_parallel_nodes(env) == 2


def test_rules_classifier_resolves_from_profile() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "orchestration_profile": OrchestrationProfile(
                classifier_kind="rules",
                intent_routes=[
                    IntentRoute(capability="echo.pipeline", keywords=["pipeline"]),
                ],
            ),
        }
    )
    classifier = resolve_nexus_task_classifier(registry, env)
    assert isinstance(classifier, RulesTaskClassifier)


def test_llm_classifier_requires_llm_adapter() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "orchestration_profile": OrchestrationProfile(classifier_kind="llm"),
        }
    )
    with pytest.raises(OrchestrationWiringError, match="llm_adapter"):
        resolve_nexus_task_classifier(registry, env)


def test_llm_classifier_resolves_with_llm_adapter() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "orchestration_profile": OrchestrationProfile(classifier_kind="llm"),
        }
    )
    classifier = resolve_nexus_task_classifier(
        registry,
        env,
        wiring_context=OrchestrationWiringContext(llm_adapter=_StubLLMAdapter()),  # type: ignore[arg-type]
    )
    assert isinstance(classifier, LlmTaskClassifier)
