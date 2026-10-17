# © Artur Czarnecki. All rights reserved.
"""PROD-Q-R1-R7-D3-R1 — UAEP CONTEXT_ASSEMBLED v4 producer contract (qualification)."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.events.payload_registry import (
    RuntimeEventPayloadError,
    assert_canonical_production_runtime_event_payload,
    runtime_event_with_payload,
)
from intergrax.runtime.events.payloads.canonical import (
    ContextAssemblyPayloadV2,
    ContextAssemblyPayloadV4,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.runtime_event_payload_policy import get_runtime_event_payload_policy
from intergrax.runtime.events.schema_guard import assert_runtime_event_schema
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = pytest.mark.gate


def _uaep_style_context_assembled_event() -> RuntimeEvent:
    """Mirror ``UaepExecutor._emit_context_assembled`` envelope (producer-only contract)."""
    ids = runtime_event_test_identity()
    typed = ContextAssemblyPayloadV4(
        node_id="node-uaep",
        context_original_chars=120,
        context_final_chars=100,
        trimmed=False,
        engine_id="default",
    )
    base = RuntimeEvent(
        task_id=ids["task_id"],
        run_id=ids["run_id"],
        attempt_id=ids["attempt_id"],
        execution_id=ids["execution_id"],
        node_id="node-uaep",
        agent_id="agent-canonical",
        tenant_id="tenant-1",
        event_type=RuntimeEventType.CONTEXT_ASSEMBLED,
        phase=ExecutionPhase.CONTEXT_BUILDING,
        correlation_id=str(ids["task_id"]),
    )
    return runtime_event_with_payload(
        base,
        typed,
        promote_fields={
            "node_id": "node-uaep",
            "engine_id": "default",
            "context_original_chars": 120,
            "context_final_chars": 100,
        },
    )


def test_context_assembled_policy_requires_context_assembly_v4() -> None:
    policy = get_runtime_event_payload_policy(RuntimeEventType.CONTEXT_ASSEMBLED)
    assert policy.schema_id == ContextAssemblyPayloadV4.schema_id


def test_uaep_style_context_assembled_v4_passes_schema_guard() -> None:
    event = _uaep_style_context_assembled_event()
    assert event.payload["payload_schema_id"] == "context_assembly.v4"
    assert_runtime_event_schema(event)
    assert_canonical_production_runtime_event_payload(event)


def test_context_assembled_rejects_legacy_v2_schema_id() -> None:
    ids = runtime_event_test_identity()
    legacy = ContextAssemblyPayloadV2(
        node_id="node-legacy",
        context_original_chars=1,
        context_final_chars=1,
    )
    event = runtime_event_with_payload(
        RuntimeEvent(
            task_id=ids["task_id"],
            run_id=ids["run_id"],
            attempt_id=ids["attempt_id"],
            execution_id=ids["execution_id"],
            event_type=RuntimeEventType.CONTEXT_ASSEMBLED,
            phase=ExecutionPhase.CONTEXT_BUILDING,
        ),
        legacy,
    )
    with pytest.raises(RuntimeEventPayloadError, match="context_assembly.v4"):
        assert_canonical_production_runtime_event_payload(event)
