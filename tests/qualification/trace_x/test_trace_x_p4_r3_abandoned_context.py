# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R3 abandoned context relation lifecycle (TXP4R3-LIFE-01..04)."""

from __future__ import annotations

import pytest

from intergrax.context.contracts import AssembledContext
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.llm.messages import ChatMessage, compute_model_facing_messages_hash
from intergrax.runtime.context_lifecycle.contracts import ModelCallExecutionScope
from intergrax.runtime.events.active_runtime_event_recorder import (
    bind_active_runtime_event_recorder,
    reset_active_runtime_event_recorder,
)
from intergrax.runtime.events.context_skill_recording import record_context_assembled_from_engine
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.llm_call_recording import record_llm_call_runtime_event
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.execution.failure_evidence.active_context import (
    ActiveExecutionEvidenceContext,
    bind_active_execution_evidence_context,
    reset_active_execution_evidence_context,
)
from intergrax.runtime.execution.failure_evidence.runtime_event_recorder import (
    RuntimeEventExecutionFailureEvidenceRecorder,
)
from intergrax.runtime.llm.model_call_attribution import (
    model_call_attribution_scope,
    peek_pending_context_assembly_event_id,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def _assembled(messages: tuple[ChatMessage, ...] = (ChatMessage(role="user", content="x"),)) -> AssembledContext:
    return AssembledContext(
        messages=messages,
        fragments_included=(),
        fragments_excluded=(),
        provenance=(),
        total_tokens=1,
        budget_tokens=100,
    )


def test_txp4r3_life_01_true_abandoned_assembly_not_visible_in_e2() -> None:
    bus = RuntimeEventBus(record_history=True)
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    messages = (ChatMessage(role="user", content="abandon"),)
    stale_ce = ""
    execution_ids: list[str] = []
    for iteration in range(2):
        execution_id = mint_execution_id()
        execution_ids.append(str(execution_id))
        id_token = bind_active_execution_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            task_id=task_id,
        )
        recorder_binding = bind_active_runtime_event_recorder(bus, tenant_id="tenant-a")
        try:
            if iteration == 0:
                record_context_assembled_from_engine(
                    bus, assembled=_assembled(messages), task_id=task_id, run_id=run_id
                )
                stale_ce = peek_pending_context_assembly_event_id()
                assert stale_ce
            else:
                with model_call_attribution_scope(messages=messages):
                    assert peek_pending_context_assembly_event_id() == ""
                    assert stale_ce
        finally:
            reset_active_runtime_event_recorder(recorder_binding)
            reset_active_execution_identity(id_token)
    assert execution_ids[0] != execution_ids[1]
    assert peek_pending_context_assembly_event_id() == ""


def test_txp4r3_life_02_stale_ce_cannot_reach_e2_llm_call() -> None:
    bus = RuntimeEventBus(record_history=True)
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    messages = (ChatMessage(role="user", content="stale"),)
    stale_ce = ""
    for iteration in range(2):
        execution_id = mint_execution_id()
        id_token = bind_active_execution_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            task_id=task_id,
        )
        recorder_binding = bind_active_runtime_event_recorder(bus, tenant_id="tenant-a")
        evidence_token = bind_active_execution_evidence_context(
            ActiveExecutionEvidenceContext(
                tenant_id="tenant-a",
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                recorder=RuntimeEventExecutionFailureEvidenceRecorder(bus),
            ),
        )
        try:
            if iteration == 0:
                record_context_assembled_from_engine(
                    bus, assembled=_assembled(messages), task_id=task_id, run_id=run_id
                )
                stale_ce = peek_pending_context_assembly_event_id()
            else:
                with model_call_attribution_scope(
                    messages=messages,
                    execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
                ):
                    with pytest.raises(ValueError, match="context_assembly_event_id required"):
                        record_llm_call_runtime_event(
                            bus,
                            task_id=task_id,
                            run_id=run_id,
                            model="m",
                            provider="openai",
                            prompt_tokens=1,
                            completion_tokens=1,
                            total_tokens=2,
                            finish_reason="stop",
                            model_input_messages_hash=compute_model_facing_messages_hash(messages),
                            execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
                        )
        finally:
            reset_active_runtime_event_recorder(recorder_binding)
            reset_active_execution_evidence_context(evidence_token)
            reset_active_execution_identity(id_token)
    llm_events = [event for event in bus.history if event.event_type == RuntimeEventType.LLM_CALL]
    assert not any(
        (event.payload or {}).get("context_assembly_event_id") == stale_ce for event in llm_events
    )


def test_txp4r3_life_03_same_execution_ce_remains_usable() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    task_id = mint_task_id()
    id_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        task_id=task_id,
    )
    bus = RuntimeEventBus(record_history=True)
    messages = (ChatMessage(role="user", content="same-exec"),)
    try:
        record_context_assembled_from_engine(bus, assembled=_assembled(messages), task_id=task_id, run_id=run_id)
        bound_ce = peek_pending_context_assembly_event_id()
        with model_call_attribution_scope(messages=messages):
            assert peek_pending_context_assembly_event_id() == bound_ce
    finally:
        reset_active_execution_identity(id_token)


def test_txp4r3_life_04_nested_scope_restoration() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    task_id = mint_task_id()
    id_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        task_id=task_id,
    )
    bus = RuntimeEventBus(record_history=True)
    messages = (ChatMessage(role="user", content="nest"),)
    opt_messages = (ChatMessage(role="user", content="opt"),)
    try:
        record_context_assembled_from_engine(bus, assembled=_assembled(messages), task_id=task_id, run_id=run_id)
        outer_ce = peek_pending_context_assembly_event_id()
        with model_call_attribution_scope(
            messages=messages,
            execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
        ):
            assert peek_pending_context_assembly_event_id() == outer_ce
            with model_call_attribution_scope(
                messages=opt_messages,
                execution_scope=ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL,
            ):
                record_context_assembled_from_engine(
                    bus, assembled=_assembled(opt_messages), task_id=task_id, run_id=run_id
                )
                inner_ce = peek_pending_context_assembly_event_id()
                assert inner_ce != outer_ce
            assert peek_pending_context_assembly_event_id() == outer_ce
    finally:
        reset_active_execution_identity(id_token)


def test_txp4r3_execution_identity_contradiction_cleared_before_scope_body() -> None:
    bus = RuntimeEventBus(record_history=True)
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    messages = (ChatMessage(role="user", content="id-mismatch"),)
    e1_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
        task_id=task_id,
    )
    record_context_assembled_from_engine(bus, assembled=_assembled(messages), task_id=task_id, run_id=run_id)
    stale_ce = peek_pending_context_assembly_event_id()
    reset_active_execution_identity(e1_token)
    e2_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
        task_id=task_id,
    )
    try:
        with model_call_attribution_scope(messages=messages):
            pending = peek_pending_context_assembly_event_id()
            assert pending == ""
            assert stale_ce
    finally:
        reset_active_execution_identity(e2_token)
