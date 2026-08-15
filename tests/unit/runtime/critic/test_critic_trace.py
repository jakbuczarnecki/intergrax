# © Artur Czarnecki. All rights reserved.

"""CRIT-V-3.6 critic trace emission tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.critic.contracts import (
    CriticAction,
    CriticLayer,
    CriticScope,
    CriticVerdict,
    LayerVerdict,
    build_critic_request,
)
from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.runtime.critic.trace import (
    CRITIC_STEP_FINAL_VERDICT,
    CRITIC_STEP_L0_FAILED,
    CriticTraceEmitter,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.trace_bridge import trace_event_to_runtime_event
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.runtime.task.task import Task

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_critic_trace_emitter_l0_fail_and_final_verdict() -> None:
    store = InMemoryRunTraceStore()
    emitter = CriticTraceEmitter(run_id="run-critic-1", trace_writer=store)
    request = build_critic_request(
        scope=CriticScope.GRAPH_FINAL,
        run_id="run-critic-1",
        agent_id="worker",
    )
    verdict = CriticVerdict(
        scope=CriticScope.GRAPH_FINAL,
        passed=False,
        layers=[
            LayerVerdict(
                layer=CriticLayer.L0_DETERMINISTIC,
                passed=False,
                score=0.0,
                errors=["empty summary"],
            ),
        ],
        recommended_action=CriticAction.FAIL,
        failure_reasons=["empty summary"],
    )

    events = emitter.emit_verdict(
        request,
        verdict,
        tenant_id="tenant-1",
        task_id="task-1",
        agent_id="worker",
        node_id="n1",
    )

    assert len(events) == 2
    assert events[0].component == TraceComponent.CRITIC
    assert events[0].step == CRITIC_STEP_L0_FAILED
    assert events[0].level == TraceLevel.ERROR
    assert events[1].step == CRITIC_STEP_FINAL_VERDICT
    assert "run-critic-1" in store._events_by_run
    assert len(store._events_by_run["run-critic-1"]) == 2


def test_critic_trace_emitter_skips_passed_l0_layer() -> None:
    emitter = CriticTraceEmitter(run_id="run-critic-2")
    request = build_critic_request(
        scope=CriticScope.NODE_PARTIAL,
        run_id="run-critic-2",
        agent_id="worker",
    )
    verdict = CriticVerdict(
        scope=CriticScope.NODE_PARTIAL,
        passed=True,
        layers=[
            LayerVerdict(layer=CriticLayer.L0_DETERMINISTIC, passed=True, score=1.0),
        ],
    )

    events = emitter.emit_verdict(
        request,
        verdict,
        tenant_id="tenant-1",
        task_id="task-1",
        agent_id="worker",
        node_id="n1",
    )

    assert events == []


def test_trace_bridge_maps_critic_l0_failed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task = Task(
        task_id=task_id,
        tenant_id="tenant-1",
        user_id="user-1",
        agent_id="worker",
        message="q",
    )
    emitter = CriticTraceEmitter(run_id=run_id)
    request = build_critic_request(
        scope=CriticScope.NODE_PARTIAL,
        run_id=run_id,
        agent_id="worker",
    )
    verdict = CriticVerdict(
        scope=CriticScope.NODE_PARTIAL,
        passed=False,
        layers=[
            LayerVerdict(
                layer=CriticLayer.L0_DETERMINISTIC,
                passed=False,
                errors=["schema mismatch"],
            ),
        ],
        failure_reasons=["schema mismatch"],
    )
    events = emitter.emit_verdict(
        request,
        verdict,
        tenant_id="tenant-1",
        task_id=task_id,
        agent_id="worker",
    )
    runtime_event = trace_event_to_runtime_event(
        events[0],
        task,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    assert runtime_event.event_type == RuntimeEventType.VALIDATION_FAILED
    assert runtime_event.phase == ExecutionPhase.VALIDATION


def test_trace_bridge_maps_critic_final_verdict_pass() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task = Task(
        task_id=task_id,
        tenant_id="tenant-1",
        user_id="user-1",
        agent_id="worker",
        message="q",
    )
    emitter = CriticTraceEmitter(run_id=run_id)
    request = build_critic_request(
        scope=CriticScope.GRAPH_FINAL,
        run_id=run_id,
        agent_id="worker",
    )
    verdict = CriticVerdict(
        scope=CriticScope.GRAPH_FINAL,
        passed=True,
        layers=[LayerVerdict(layer=CriticLayer.L0_DETERMINISTIC, passed=True, score=1.0)],
    )
    events = emitter.emit_verdict(
        request,
        verdict,
        tenant_id="tenant-1",
        task_id=task_id,
        agent_id="worker",
    )
    final_evt = next(evt for evt in events if evt.step == CRITIC_STEP_FINAL_VERDICT)
    runtime_event = trace_event_to_runtime_event(
        final_evt,
        task,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    assert runtime_event.event_type == RuntimeEventType.STEP_COMPLETED
    from intergrax.runtime.events.phase_coverage import phase_for_event

    assert runtime_event.phase == phase_for_event(RuntimeEventType.STEP_COMPLETED)
