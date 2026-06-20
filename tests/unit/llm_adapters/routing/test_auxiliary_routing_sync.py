# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import pytest

from intergrax.llm_adapters.routing.context_bridge import LLMRoutingRuntimeSnapshot
from intergrax.llm_adapters.routing.contracts import RoutingContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.routing_snapshot_sync import sync_routing_for_task_llm_call
from intergrax.runtime.nexus.llm_task_classifier import LlmTaskClassifier
from intergrax.runtime.nexus.planning.nexus_plan_bridge import build_nexus_plan_unified
from intergrax.runtime.nexus.planning.task_planner import TaskPlanner
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from echo.echo_agent import EchoAgent
from testing_support.builder import FakeLLMAdapter


@pytest.mark.unit
@pytest.mark.gate
def test_plan_bridge_refreshes_routing_snapshot_before_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float | None] = []
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        llm_routing_snapshot=LLMRoutingRuntimeSnapshot(metadata={}),
        llm_routing_context=RoutingContext(budget_remaining_ratio=0.5),
    )

    def _fake_refresh(cfg, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs.get("metadata", {}).get("budget_remaining_ratio"))
        snapshot = cfg.llm_routing_snapshot
        assert snapshot is not None
        snapshot.metadata["refreshed"] = True
        return RoutingContext()

    monkeypatch.setattr(
        "intergrax.runtime.nexus.context.routing_snapshot_sync.refresh_config_routing_snapshot",
        _fake_refresh,
    )

    registry = AgentRegistry()
    registry.register(EchoAgent())
    agent_id = registry.list_agent_ids()[0]
    llm = FakeLLMAdapter(
        fixed_text=json.dumps(
            {"steps": [{"agent_id": agent_id, "description": "step", "depends_on": []}]}
        )
    )
    task = Task(
        tenant_id="t",
        user_id="u",
        message="hi",
        context=TaskContext(),
        metadata={
            "nexus_runtime_config": config,
            "budget_remaining_ratio": 0.25,
        },
    )
    build_nexus_plan_unified(
        task,
        registry,
        llm,
        fallback=TaskPlanner(),
        prompt_text="plan",
    )
    assert calls == [0.25]
    assert config.llm_routing_snapshot.metadata.get("refreshed") is True


@pytest.mark.unit
@pytest.mark.gate
def test_task_classifier_refreshes_routing_snapshot_before_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        llm_routing_snapshot=LLMRoutingRuntimeSnapshot(metadata={}),
        llm_routing_context=RoutingContext(),
    )

    def _fake_refresh(cfg, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("refresh")
        return RoutingContext()

    monkeypatch.setattr(
        "intergrax.runtime.nexus.context.routing_snapshot_sync.refresh_config_routing_snapshot",
        _fake_refresh,
    )

    registry = AgentRegistry()
    registry.register(EchoAgent())
    from intergrax.applications.contracts.intent_route import IntentRoute

    classifier = LlmTaskClassifier(
        registry,
        FakeLLMAdapter(
            fixed_text=json.dumps(
                {"capability": "echo.basic", "confidence": 0.9, "rationale": "test"}
            )
        ),
        intent_routes=[IntentRoute(capability="echo.basic", keywords=["hi"])],
    )
    task = Task(
        tenant_id="t",
        user_id="u",
        message="hi",
        context=TaskContext(),
        metadata={"nexus_runtime_config": config},
    )
    classifier.classify(task)
    assert calls == ["refresh"]


@pytest.mark.unit
@pytest.mark.gate
def test_sync_routing_for_task_llm_call_noops_without_config() -> None:
    task = Task(tenant_id="t", user_id="u", message="hi", context=TaskContext())
    sync_routing_for_task_llm_call(task)
