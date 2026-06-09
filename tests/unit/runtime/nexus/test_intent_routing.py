# © Artur Czarnecki. All rights reserved.

"""Intent routing and rules classifier (ORCH-CONFIG.1)."""

from __future__ import annotations

import pytest

from intergrax.applications.contracts.intent_route import IntentRoute
from intergrax.runtime.nexus.intent_routing import apply_intent_routes
from intergrax.runtime.nexus.rules_task_classifier import RulesTaskClassifier
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from dispute_analyst.dispute_analyst_agent import DisputeAnalystAgent
from dispute_scenario.dispute_scenario_agent import DisputeScenarioAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_apply_intent_routes_matches_polish_keywords() -> None:
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="Mamy podwykonawcę który domaga się zapłaty za wadliwe prace. Jak odpisać?",
        context=TaskContext(),
    )
    routes = [
        IntentRoute(capability="dispute.pipeline", keywords=["podwykonaw", "odpisa"]),
    ]
    routed = apply_intent_routes(task, routes)
    assert routed.context.capability == "dispute.pipeline"


def test_apply_intent_routes_preserves_explicit_capability() -> None:
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(capability="dispute.intake"),
    )
    routes = [IntentRoute(capability="dispute.pipeline", keywords=["hello"])]
    routed = apply_intent_routes(task, routes)
    assert routed.context.capability == "dispute.intake"


def test_rules_classifier_routes_before_classification() -> None:
    registry = AgentRegistry()
    registry.register(DisputeAnalystAgent())
    registry.register(DisputeScenarioAgent())
    classifier = RulesTaskClassifier(
        registry,
        intent_routes=[
            IntentRoute(capability="dispute.pipeline", keywords=["podwykonaw"]),
        ],
    )
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="Podwykonawca domaga się zapłaty",
        context=TaskContext(),
    )
    classified = classifier.classify(task)
    assert classified.context.capability == "dispute.pipeline"
    assert classified.classification == "multi_agent"
