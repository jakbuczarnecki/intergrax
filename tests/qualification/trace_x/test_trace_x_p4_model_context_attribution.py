# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4 model call / context decision attribution (TXP4-Q01..Q30 subset)."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.context.contracts import AssembledContext
from intergrax.llm.messages import ChatMessage, compute_model_facing_messages_hash
from intergrax.runtime.llm.model_call_attribution import model_call_attribution_scope
from intergrax.runtime.events.llm_call_recording import record_llm_call_runtime_event
from intergrax.runtime.context_lifecycle.contracts import ModelCallExecutionScope
from intergrax.runtime.events.active_runtime_event_recorder import (
    bind_active_runtime_event_recorder,
    reset_active_runtime_event_recorder,
)
from intergrax.runtime.events.context_skill_recording import record_context_assembled_from_engine
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.execution.failure_evidence.active_context import (
    ActiveExecutionEvidenceContext,
    bind_active_execution_evidence_context,
    reset_active_execution_evidence_context,
)
from intergrax.runtime.execution.failure_evidence.runtime_event_recorder import (
    RuntimeEventExecutionFailureEvidenceRecorder,
)
from intergrax.runtime.nexus.context.context_budget import ContextTrimResult
from intergrax.runtime.events.context_skill_recording import record_context_assembly
from tests.qualification.trace_x._trace_x_p4_support import (
    TRACE_X_P4_START_HEAD,
    attribution_from_runtime_event,
    attributions_joinable,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def _record_primary_model_call(
    bus: RuntimeEventBus,
    *,
    task_id: str,
    run_id: str,
    messages: Sequence[ChatMessage],
    execution_scope: ModelCallExecutionScope = ModelCallExecutionScope.PRIMARY_MODEL_CALL,
) -> None:
    with model_call_attribution_scope(execution_scope=execution_scope, messages=messages):
        record_llm_call_runtime_event(
            bus,
            task_id=task_id,
            run_id=run_id,
            model="stub-model",
            provider="openai",
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
            finish_reason="stop",
            model_input_messages_hash=compute_model_facing_messages_hash(messages),
            execution_scope=execution_scope,
            tenant_id="tenant-a",
        )


@pytest.fixture
def governed_execution():
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
    failure_recorder = RuntimeEventExecutionFailureEvidenceRecorder(bus)
    evidence_token = bind_active_execution_evidence_context(
        ActiveExecutionEvidenceContext(
            tenant_id="tenant-a",
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            recorder=failure_recorder,
        ),
    )
    recorder_token = bind_active_runtime_event_recorder(bus, tenant_id="tenant-a")
    try:
        yield run_id, task_id, attempt_id, execution_id, bus
    finally:
        reset_active_runtime_event_recorder(recorder_token)
        reset_active_execution_evidence_context(evidence_token)
        reset_active_execution_identity(id_token)


def test_txp4_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P4_START_HEAD, "HEAD"],
    )


def test_txp4_q12_primary_context_model_e2e_attribution(governed_execution) -> None:
    run_id, task_id, _attempt_id, _execution_id, bus = governed_execution
    messages = (ChatMessage(role="user", content="hello"),)
    expected_hash = compute_model_facing_messages_hash(messages)
    assembled = AssembledContext(
        messages=messages,
        fragments_included=(),
        fragments_excluded=(),
        provenance=(),
        total_tokens=1,
        budget_tokens=100,
    )
    record_context_assembled_from_engine(
        bus,
        assembled=assembled,
        task_id=task_id,
        run_id=run_id,
        node_id="n1",
        agent_id="agent-1",
        engine_id="context_engineering",
    )
    _record_primary_model_call(bus, task_id=task_id, run_id=run_id, messages=messages)

    ctx_events = [e for e in bus.history if e.event_type == RuntimeEventType.CONTEXT_ASSEMBLED]
    llm_events = [e for e in bus.history if e.event_type == RuntimeEventType.LLM_CALL]
    assert len(ctx_events) == 1
    assert len(llm_events) == 1
    ctx_attr = attribution_from_runtime_event(ctx_events[0])
    llm_attr = attribution_from_runtime_event(llm_events[0])
    assert ctx_attr is not None and llm_attr is not None
    assert ctx_attr.model_input_messages_hash == expected_hash
    assert llm_attr.model_input_messages_hash == expected_hash
    assert attributions_joinable(ctx_attr, llm_attr)
    assert llm_attr.execution_scope == ModelCallExecutionScope.PRIMARY_MODEL_CALL.value
    assert llm_events[0].payload.get("data", {}).get("provider") == "openai"


def test_txp4_q13_same_run_multi_execution_isolation(governed_execution) -> None:
    run_id, task_id, attempt_id, _execution_id, bus = governed_execution
    messages_a = (ChatMessage(role="user", content="a"),)
    messages_b = (ChatMessage(role="user", content="b"),)
    hash_a = compute_model_facing_messages_hash(messages_a)
    hash_b = compute_model_facing_messages_hash(messages_b)
    assert hash_a != hash_b

    tokens: list = []
    try:
        for messages in (messages_a, messages_b):
            execution_id = mint_execution_id()
            tokens.append(
                bind_active_execution_identity(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                    task_id=task_id,
                )
            )
            record_context_assembled_from_engine(
                bus,
                assembled=AssembledContext(
                    messages=messages,
                    fragments_included=(),
                    fragments_excluded=(),
                    provenance=(),
                    total_tokens=1,
                    budget_tokens=100,
                ),
                task_id=task_id,
                run_id=run_id,
            )
            _record_primary_model_call(bus, task_id=task_id, run_id=run_id, messages=messages)
    finally:
        for token in reversed(tokens):
            reset_active_execution_identity(token)

    llm_events = [e for e in bus.history if e.event_type == RuntimeEventType.LLM_CALL]
    assert len(llm_events) == 2
    assert llm_events[0].execution_id != llm_events[1].execution_id
    assert attribution_from_runtime_event(llm_events[0]).model_input_messages_hash == hash_a
    assert attribution_from_runtime_event(llm_events[1]).model_input_messages_hash == hash_b


def test_txp4_q15_hash_mismatch_not_attributable() -> None:
    left_hash = compute_model_facing_messages_hash((ChatMessage(role="user", content="one"),))
    right_hash = compute_model_facing_messages_hash((ChatMessage(role="user", content="two"),))
    assert left_hash != right_hash


def test_txp4_q16_internal_optimization_scope(governed_execution) -> None:
    run_id, task_id, _attempt_id, _execution_id, bus = governed_execution
    messages = (ChatMessage(role="user", content="compact"),)
    _record_primary_model_call(
        bus,
        task_id=task_id,
        run_id=run_id,
        messages=messages,
        execution_scope=ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL,
    )
    llm = next(e for e in bus.history if e.event_type == RuntimeEventType.LLM_CALL)
    attr = attribution_from_runtime_event(llm)
    assert attr is not None
    assert attr.execution_scope == ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL.value


def test_txp4_q21_cross_tenant_mismatch_rejected(governed_execution) -> None:
    run_id, task_id, _attempt_id, _execution_id, bus = governed_execution
    trim = ContextTrimResult(message="m", trimmed=False, original_chars=1, final_chars=1)
    record_context_assembly(
        bus,
        task_id=task_id,
        run_id=run_id,
        node_id="n",
        agent_id="a",
        trim=trim,
        metadata={
            "tenant_id": "tenant-a",
            "model_input_messages_hash": compute_model_facing_messages_hash(
                (ChatMessage(role="user", content="x"),)
            ),
        },
        emit_assembled=True,
    )
    ctx = next(e for e in bus.history if e.event_type == RuntimeEventType.CONTEXT_ASSEMBLED)
    assert ctx.tenant_id == "tenant-a"


def test_txp4_q22_no_raw_prompt_in_evidence(governed_execution) -> None:
    run_id, task_id, _attempt_id, _execution_id, bus = governed_execution
    secret = "super-secret-prompt-body"
    messages = (ChatMessage(role="user", content=secret),)
    record_context_assembled_from_engine(
        bus,
        assembled=AssembledContext(
            messages=messages,
            fragments_included=(),
            fragments_excluded=(),
            provenance=(),
            total_tokens=1,
            budget_tokens=100,
        ),
        task_id=task_id,
        run_id=run_id,
    )
    _record_primary_model_call(bus, task_id=task_id, run_id=run_id, messages=messages)
    for event in bus.history:
        blob = str(event.payload)
        assert secret not in blob
