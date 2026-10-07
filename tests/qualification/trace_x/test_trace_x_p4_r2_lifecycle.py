# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R2 context relation lifecycle gates (TXP4R2-LIFE-01..06)."""

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
from intergrax.runtime.events.llm_call_recording import (
    maybe_record_llm_call_from_usage_end,
    record_llm_call_runtime_event,
)
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


@pytest.fixture
def bare_execution():
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
    try:
        yield run_id, task_id, bus
    finally:
        reset_active_execution_identity(id_token)


def test_txp4r2_life_01_successful_model_call_clears_context_ref(bare_execution) -> None:
    run_id, task_id, bus = bare_execution
    messages = (ChatMessage(role="user", content="ok"),)
    record_context_assembled_from_engine(bus, assembled=_assembled(messages), task_id=task_id, run_id=run_id)
    assert peek_pending_context_assembly_event_id()
    with model_call_attribution_scope(messages=messages):
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
    assert peek_pending_context_assembly_event_id() == ""


def test_txp4r2_life_02_provider_failure_clears_context_ref(bare_execution) -> None:
    run_id, task_id, bus = bare_execution
    messages = (ChatMessage(role="user", content="fail"),)
    record_context_assembled_from_engine(bus, assembled=_assembled(messages), task_id=task_id, run_id=run_id)
    with pytest.raises(RuntimeError):
        with model_call_attribution_scope(messages=messages):
            raise RuntimeError("provider failure")
    assert peek_pending_context_assembly_event_id() == ""


def test_txp4r2_life_03_no_recorder_path_cannot_leak_context_ref(bare_execution) -> None:
    run_id, task_id, bus = bare_execution
    messages = (ChatMessage(role="user", content="nr"),)
    record_context_assembled_from_engine(bus, assembled=_assembled(messages), task_id=task_id, run_id=run_id)
    with model_call_attribution_scope(messages=messages):
        maybe_record_llm_call_from_usage_end(
            run_id=run_id,
            provider="openai",
            model="m",
            input_tokens=1,
            output_tokens=1,
            success=True,
            finish_reason="stop",
            model_input_messages_hash=compute_model_facing_messages_hash(messages),
        )
    assert peek_pending_context_assembly_event_id() == ""


def test_txp4r2_life_04_missing_evidence_context_cannot_leak_context_ref(bare_execution) -> None:
    run_id, task_id, bus = bare_execution
    messages = (ChatMessage(role="user", content="me"),)
    recorder_binding = bind_active_runtime_event_recorder(bus, tenant_id="tenant-a")
    record_context_assembled_from_engine(bus, assembled=_assembled(messages), task_id=task_id, run_id=run_id)
    try:
        with model_call_attribution_scope(messages=messages):
            maybe_record_llm_call_from_usage_end(
                run_id=run_id,
                provider="openai",
                model="m",
                input_tokens=1,
                output_tokens=1,
                success=True,
                finish_reason="stop",
                model_input_messages_hash=compute_model_facing_messages_hash(messages),
            )
    finally:
        reset_active_runtime_event_recorder(recorder_binding)
    assert peek_pending_context_assembly_event_id() == ""


def test_txp4r2_life_05_sequential_executions_cannot_inherit_stale_ce() -> None:
    bus = RuntimeEventBus(record_history=True)
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    messages = (ChatMessage(role="user", content="seq"),)
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
                with model_call_attribution_scope(messages=messages):
                    assert peek_pending_context_assembly_event_id() == ""
        finally:
            reset_active_runtime_event_recorder(recorder_binding)
            reset_active_execution_evidence_context(evidence_token)
            reset_active_execution_identity(id_token)
    assert peek_pending_context_assembly_event_id() == ""


def test_txp4r2_life_06_nested_model_attribution_scope_restores_outer_relation(bare_execution) -> None:
    run_id, task_id, bus = bare_execution
    messages = (ChatMessage(role="user", content="nest"),)
    opt_messages = (ChatMessage(role="user", content="opt"),)
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
    assert peek_pending_context_assembly_event_id() == ""
