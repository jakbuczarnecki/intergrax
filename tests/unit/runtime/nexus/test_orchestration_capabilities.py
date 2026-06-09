# © Artur Czarnecki. All rights reserved.

"""Orchestration capability tokens (ORCH-CONFIG)."""

from __future__ import annotations

import pytest

from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec, GraphNode
from intergrax.runtime.nexus.orchestration_capabilities import (
    is_orchestration_capability,
    orchestration_capabilities_from_graph_spec,
)
from intergrax.runtime.nexus.task_classifier import ClassifyingTaskClassifier
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from echo.echo_agent import EchoAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_orchestration_capability_from_trigger_list() -> None:
    spec = ApplicationGraphSpec(
        nodes=[GraphNode(agent_id="a")],
        trigger_capabilities=["product.pipeline"],
    )
    triggers = orchestration_capabilities_from_graph_spec(spec)
    assert is_orchestration_capability(
        "product.pipeline",
        trigger_capabilities=triggers,
    )


def test_orchestration_capability_from_pipeline_suffix() -> None:
    assert is_orchestration_capability(
        "lab.pipeline",
        trigger_capabilities=frozenset(),
        pipeline_capability_suffix=".pipeline",
    )


def test_classifier_accepts_orchestration_token_without_registry_agent() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    classifier = ClassifyingTaskClassifier(
        registry,
        orchestration_trigger_capabilities=frozenset({"acceptance.harness.pipeline"}),
    )
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(capability="acceptance.harness.pipeline"),
    )
    classified = classifier.classify(task)
    assert classified.classification == "capability_routed"
