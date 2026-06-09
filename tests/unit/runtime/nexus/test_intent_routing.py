# © Artur Czarnecki. All rights reserved.

"""Intent routing and rules classifier (ORCH-CONFIG.1)."""

from __future__ import annotations

import pytest

from intergrax.applications.contracts.intent_route import IntentRoute
from intergrax.runtime.nexus.intent_routing import apply_intent_routes
from intergrax.runtime.nexus.rules_task_classifier import RulesTaskClassifier
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from echo.echo_agent import EchoAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_PIPELINE = "acceptance.harness.pipeline"


def test_apply_intent_routes_matches_keywords() -> None:
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="Keyword trigger for pipeline routing in harness simulation",
        context=TaskContext(),
    )
    routes = [IntentRoute(capability=_PIPELINE, keywords=["pipeline", "routing"])]
    apply_intent_routes(task, routes)
    assert task.context.capability == _PIPELINE


def test_apply_intent_routes_preserves_explicit_capability() -> None:
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(capability="echo.basic"),
    )
    routes = [IntentRoute(capability=_PIPELINE, keywords=["hello"])]
    apply_intent_routes(task, routes)
    assert task.context.capability == "echo.basic"


def test_rules_classifier_routes_orchestration_capability() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    classifier = RulesTaskClassifier(
        registry,
        intent_routes=[IntentRoute(capability=_PIPELINE, keywords=["podwykonaw"])],
        orchestration_trigger_capabilities=frozenset({_PIPELINE}),
    )
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="Podwykonawca domaga się zapłaty",
        context=TaskContext(),
    )
    classified = classifier.classify(task)
    assert classified.context.capability == _PIPELINE
    assert classified.classification == "capability_routed"
    assert classified.runtime.classification.classifier_source == "rules"
    assert classified.runtime.classification.confidence == 1.0
