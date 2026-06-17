# © Artur Czarnecki. All rights reserved.

"""LLM task classifier with rules fallback (COG-3.2)."""

from __future__ import annotations

import json

import pytest

from intergrax.applications.contracts.intent_route import IntentRoute
from intergrax.runtime.nexus.llm_task_classifier import LlmTaskClassifier
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from echo.echo_agent import EchoAgent
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_PIPELINE = "acceptance.harness.pipeline"


def test_llm_classifier_infers_capability_with_trace() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    llm = FakeLLMAdapter(
        fixed_text=json.dumps(
            {
                "capability": _PIPELINE,
                "confidence": 0.92,
                "rationale": "payment dispute narrative",
            }
        )
    )
    classifier = LlmTaskClassifier(
        registry,
        llm,
        intent_routes=[IntentRoute(capability=_PIPELINE, keywords=["unused"])],
        orchestration_trigger_capabilities=frozenset({_PIPELINE}),
    )
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="Subcontractor demands payment for defective work",
        context=TaskContext(),
    )
    classified = classifier.classify(task)
    assert classified.context.capability == _PIPELINE
    assert classified.classification == "capability_routed"
    assert classified.runtime.classification.classifier_source == "llm"
    assert classified.runtime.classification.confidence == pytest.approx(0.92)
    assert "payment" in (classified.runtime.classification.rationale or "")


def test_llm_classifier_falls_back_to_rules_on_parse_failure() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    llm = FakeLLMAdapter(fixed_text="not json")
    classifier = LlmTaskClassifier(
        registry,
        llm,
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
    assert classified.runtime.classification.classifier_source == "rules"
    assert classified.metadata.get("reasoning_failure_kind") == "classifier_fallback"
