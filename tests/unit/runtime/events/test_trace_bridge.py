# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id, mint_task_id
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.phase_coverage import phase_for_event
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.trace_bridge import (
    trace_bridge_subject_from_tags,
    trace_event_to_runtime_event,
)
from intergrax.runtime.nexus.tracing.adapters.core_llm_call_recorded import CoreLLMCallRecordedDiagV1
from intergrax.runtime.nexus.tracing.adapters.llm_routing_attempt import (
    LLMRoutingAttemptDiagV1,
    LLMRoutingRuleDiagV1,
)
from intergrax.runtime.nexus.tracing.adapters.model_catalog_miss import (
    ModelCatalogMissTraceDiagV1,
)
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel
from intergrax.runtime.task.task import Task, TaskState

pytestmark = pytest.mark.gate


def _trace_identity() -> tuple[str, str, str, str]:
    return mint_task_id(), mint_run_id(), mint_attempt_id(), mint_execution_id()


def test_phase_for_event_matches_trace_bridge_retry() -> None:
    task_id, run_id, attempt_id, execution_id = _trace_identity()
    task = Task(
        task_id=task_id,
        tenant_id="tenant",
        user_id="user",
        agent_id="agent",
        message="q",
    )
    trace = TraceEvent(
        event_id="e1",
        run_id=run_id,
        seq=1,
        ts_utc="2026-06-01T00:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.PLANNER,
        step="retry",
        message="retry attempt 1",
        tags={"task_id": task_id, "task_state": TaskState.RUNNING.value},
    )
    event = trace_event_to_runtime_event(
        trace,
        task,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    assert event.event_type == RuntimeEventType.RETRY_STARTED
    assert event.phase == phase_for_event(RuntimeEventType.RETRY_STARTED)
    assert event.phase == ExecutionPhase.RETRY_HANDLING


def _task_with_identity() -> tuple[Task, str, str, str, str]:
    task_id, run_id, attempt_id, execution_id = _trace_identity()
    task = Task(
        task_id=task_id,
        tenant_id="tenant",
        user_id="user",
        agent_id="agent",
        message="q",
    )
    return task, task_id, run_id, attempt_id, execution_id


def test_trace_bridge_maps_tool_invocation_start() -> None:
    task, task_id, run_id, attempt_id, execution_id = _task_with_identity()
    trace = TraceEvent(
        event_id="tool-1",
        run_id=run_id,
        seq=3,
        ts_utc="2026-06-07T10:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.TOOLS,
        step="tool_invocation_start",
        message="invoke rag.retrieve",
        tags={"task_id": task_id, "tool_name": "rag.retrieve"},
    )
    event = trace_event_to_runtime_event(
        trace,
        task,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        payload_dict={"tool_name": "rag.retrieve"},
    )
    assert event.event_type == RuntimeEventType.TOOL_REQUESTED
    assert event.payload["tool_name"] == "rag.retrieve"
    assert event.payload["trace_seq"] == 3


def test_trace_bridge_maps_llm_call_recorded_schema() -> None:
    task, task_id, run_id, attempt_id, execution_id = _task_with_identity()
    trace = TraceEvent(
        event_id="llm-1",
        run_id=run_id,
        seq=4,
        ts_utc="2026-06-07T10:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.ENGINE,
        step="core_llm",
        message="call recorded",
        tags={"task_id": task_id},
    )
    payload = {
        "model": "gpt-test",
        "provider": "openai",
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
        "finish_reason": "stop",
        "response_id": "resp-1",
        "has_refusal": False,
        "has_tool_calls": False,
    }
    event = trace_event_to_runtime_event(
        trace,
        task,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        payload_schema_id=CoreLLMCallRecordedDiagV1.schema_id(),
        payload_dict=payload,
    )
    assert event.event_type == RuntimeEventType.LLM_CALL
    assert event.payload["model"] == "gpt-test"
    assert event.payload["total_tokens"] == 16


def test_trace_bridge_maps_llm_routing_attempt_schema() -> None:
    task, task_id, run_id, attempt_id, execution_id = _task_with_identity()
    trace = TraceEvent(
        event_id="llm-route-1",
        run_id=run_id,
        seq=5,
        ts_utc="2026-06-17T10:00:00Z",
        level=TraceLevel.WARNING,
        component=TraceComponent.ENGINE,
        step="llm_routing_attempt",
        message="LLM profile failover attempt recorded.",
        tags={"task_id": task_id},
    )
    payload = {
        "profile_id": "openai:gpt-4o",
        "provider": "openai",
        "model": "gpt-4o",
        "error": "RuntimeError: rate limited",
        "profile_index": 0,
    }
    event = trace_event_to_runtime_event(
        trace,
        task,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        payload_schema_id=LLMRoutingAttemptDiagV1.schema_id(),
        payload_dict=payload,
    )
    assert event.event_type == RuntimeEventType.LLM_CALL
    assert event.payload["model"] == "gpt-4o"


def test_trace_bridge_maps_llm_routing_rule_schema() -> None:
    task, task_id, run_id, attempt_id, execution_id = _task_with_identity()
    trace = TraceEvent(
        event_id="llm-rule-1",
        run_id=run_id,
        seq=6,
        ts_utc="2026-06-19T10:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.ENGINE,
        step="llm_routing_rule",
        message="LLM routing rule evaluation recorded.",
        tags={"task_id": task_id},
    )
    payload = {
        "matched_rule_id": "builtin.budget_below",
        "routing_reason": "rule:builtin.budget_below",
        "profile_id": "vllm:meta-llama/Llama-3.1-8B",
        "provider": "vllm",
        "model": "meta-llama/Llama-3.1-8B",
        "policy_route_hint": None,
    }
    event = trace_event_to_runtime_event(
        trace,
        task,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        payload_schema_id=LLMRoutingRuleDiagV1.schema_id(),
        payload_dict=payload,
    )
    assert event.event_type == RuntimeEventType.LLM_CALL
    assert event.payload["model"] == "meta-llama/Llama-3.1-8B"


def test_trace_bridge_maps_llm_catalog_miss_schema() -> None:
    task, task_id, run_id, attempt_id, execution_id = _task_with_identity()
    trace = TraceEvent(
        event_id="llm-miss-1",
        run_id=run_id,
        seq=7,
        ts_utc="2026-06-19T10:00:00Z",
        level=TraceLevel.WARNING,
        component=TraceComponent.ENGINE,
        step="llm_catalog_miss",
        message="Model catalog miss — conservative context window default applied.",
        tags={"task_id": task_id},
    )
    payload = {
        "provider_slug": "openrouter",
        "model_id": "vendor/unknown",
        "resolved_tokens": 8192,
        "resolution_tier": "fallback_default",
        "run_id": run_id,
    }
    event = trace_event_to_runtime_event(
        trace,
        task,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        payload_schema_id=ModelCatalogMissTraceDiagV1.schema_id(),
        payload_dict=payload,
    )
    assert event.event_type == RuntimeEventType.LLM_CALL
    assert event.payload["model"] == "vendor/unknown"
    assert event.payload["resolution_tier"] == "fallback_default"


def test_trace_bridge_ignores_noncanonical_task_id_tag() -> None:
    task, task_id, run_id, attempt_id, execution_id = _task_with_identity()
    trace = TraceEvent(
        event_id="diag-1",
        run_id=run_id,
        seq=8,
        ts_utc="2026-06-19T10:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.ENGINE,
        step="local_indexer_step",
        message="index diagnostic",
        tags={"task_id": "local_indexer", "tenant_id": "tenant"},
    )
    event = trace_event_to_runtime_event(
        trace,
        task,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    assert event.task_id == task_id


def test_trace_bridge_rejects_malformed_supplied_run_id() -> None:
    task, task_id, _, attempt_id, execution_id = _task_with_identity()
    trace = TraceEvent(
        event_id="malformed-run",
        run_id="run-bad",
        seq=9,
        ts_utc="2026-06-19T10:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.ENGINE,
        step="diag",
        message="bad run",
        tags={"task_id": task_id},
    )
    with pytest.raises(ValueError, match="malformed run_id"):
        trace_event_to_runtime_event(
            trace,
            task,
            run_id="run-bad",
            attempt_id=attempt_id,
            execution_id=execution_id,
        )


def test_trace_bridge_does_not_mint_task_id_from_malformed_tag() -> None:
    task_id, run_id, attempt_id, execution_id = _trace_identity()
    task = Task(
        task_id=task_id,
        tenant_id="tenant",
        user_id="user",
        agent_id="agent",
        message="q",
    )
    trace = TraceEvent(
        event_id="malformed-task",
        run_id=run_id,
        seq=10,
        ts_utc="2026-06-19T10:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.ENGINE,
        step="diag",
        message="bad task tag only",
        tags={"task_id": "not-canonical"},
    )
    with pytest.raises(ValueError, match="TaskId must start with"):
        trace_event_to_runtime_event(
            trace,
            trace_bridge_subject_from_tags(
                tenant_id="tenant",
                task_id="also-not-canonical",
            ),
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )


def test_trace_bridge_does_not_use_run_id_as_task_id() -> None:
    from intergrax.runtime.events.trace_bridge import TraceBridgeSubjectView

    task_id, run_id, attempt_id, execution_id = _trace_identity()
    trace = TraceEvent(
        event_id="missing-task",
        run_id=run_id,
        seq=11,
        ts_utc="2026-06-19T10:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.ENGINE,
        step="diag",
        message="no task",
        tags={"task_id": task_id},
    )
    with pytest.raises(ValueError, match="TaskId must start with"):
        trace_event_to_runtime_event(
            trace,
            TraceBridgeSubjectView(tenant_id="tenant", task_id=run_id),
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )


def test_trace_bridge_requires_attempt_id_without_active_identity() -> None:
    task, task_id, run_id, _attempt_id, execution_id = _task_with_identity()
    trace = TraceEvent(
        event_id="missing-attempt",
        run_id=run_id,
        seq=12,
        ts_utc="2026-06-19T10:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.ENGINE,
        step="diag",
        message="no attempt",
        tags={"task_id": task_id},
    )
    with pytest.raises(RuntimeError, match="explicit attempt_id required"):
        trace_event_to_runtime_event(trace, task, run_id=run_id)


def test_trace_bridge_requires_execution_id_without_active_identity() -> None:
    task, task_id, run_id, attempt_id, _execution_id = _task_with_identity()
    trace = TraceEvent(
        event_id="missing-execution",
        run_id=run_id,
        seq=13,
        ts_utc="2026-06-19T10:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.ENGINE,
        step="diag",
        message="no execution",
        tags={"task_id": task_id},
    )
    with pytest.raises(RuntimeError, match="explicit execution_id required"):
        trace_event_to_runtime_event(trace, task, run_id=run_id, attempt_id=attempt_id)


def test_trace_bridge_subject_requires_tenant_id() -> None:
    with pytest.raises(ValueError, match="tenant_id is required"):
        trace_bridge_subject_from_tags(
            tenant_id="",
            task_id=mint_task_id(),
        )
