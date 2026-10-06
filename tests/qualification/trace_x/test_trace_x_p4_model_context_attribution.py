# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4 / P4-R1 model call / context decision attribution."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

import pytest

from intergrax.context.contracts import (
    AssembledContext,
    ContextPolicyDecision,
    ContextPolicyReasonCode,
    ContextPolicyStage,
)
from intergrax.context.tracking.decision_evidence import (
    build_context_decision_evidence_canonical,
    compute_context_decision_evidence_fingerprint,
)
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
from intergrax.runtime.events.payloads.canonical import LlmCallPayloadV3
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.execution.failure_evidence.active_context import (
    ActiveExecutionEvidenceContext,
    bind_active_execution_evidence_context,
    reset_active_execution_evidence_context,
)
from intergrax.runtime.execution.failure_evidence.runtime_event_recorder import (
    RuntimeEventExecutionFailureEvidenceRecorder,
)
from intergrax.runtime.llm.model_call_attribution import model_call_attribution_scope
from intergrax.runtime.llm.model_context_attribution import recompute_context_decision_fingerprint
from tests.qualification.trace_x._trace_x_p4_support import (
    TRACE_X_P4_START_HEAD,
    attribute_model_call_to_context,
    parse_context_assembly_payload,
    parse_llm_call_payload,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def _record_primary_model_call(
    bus: RuntimeEventBus,
    *,
    task_id: str,
    run_id: str,
    messages: Sequence[ChatMessage],
    execution_scope: ModelCallExecutionScope = ModelCallExecutionScope.PRIMARY_MODEL_CALL,
    tenant_id: str = "tenant-a",
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
            tenant_id=tenant_id,
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
    recorder_binding = bind_active_runtime_event_recorder(bus, tenant_id="tenant-a")
    try:
        yield run_id, task_id, attempt_id, execution_id, bus
    finally:
        reset_active_runtime_event_recorder(recorder_binding)
        reset_active_execution_evidence_context(evidence_token)
        reset_active_execution_identity(id_token)


def test_txp4_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P4_START_HEAD, "HEAD"],
    )


def test_txp4r1_q08_invalid_execution_scope_rejected() -> None:
    with pytest.raises(ValueError):
        LlmCallPayloadV3.model_validate(
            {
                "model": "m",
                "provider": "openai",
                "model_input_messages_hash": "abc",
                "execution_scope": "anything",
            }
        )


def test_txp4r1_q11_same_messages_different_decision_fingerprint() -> None:
    messages = (ChatMessage(role="user", content="same-body"),)
    base = AssembledContext(
        messages=messages,
        fragments_included=(),
        fragments_excluded=(),
        provenance=(),
        total_tokens=1,
        budget_tokens=100,
    )
    decision_a = ContextPolicyDecision(
        stage=ContextPolicyStage.EXACT_DEDUP,
        strategy_id="strategy-a",
        input_fragment_ids=("f1",),
        output_fragment_ids=("f1",),
        reason_code=ContextPolicyReasonCode.EXACT_DUPLICATE_CONTENT,
    )
    decision_b = ContextPolicyDecision(
        stage=ContextPolicyStage.EXACT_DEDUP,
        strategy_id="strategy-b",
        input_fragment_ids=("f1",),
        output_fragment_ids=("f1",),
        reason_code=ContextPolicyReasonCode.EXACT_DUPLICATE_CONTENT,
    )
    assembly_a = AssembledContext(
        messages=base.messages,
        fragments_included=base.fragments_included,
        fragments_excluded=base.fragments_excluded,
        provenance=base.provenance,
        total_tokens=base.total_tokens,
        budget_tokens=base.budget_tokens,
        policy_decisions=(decision_a,),
    )
    assembly_b = AssembledContext(
        messages=base.messages,
        fragments_included=base.fragments_included,
        fragments_excluded=base.fragments_excluded,
        provenance=base.provenance,
        total_tokens=base.total_tokens,
        budget_tokens=base.budget_tokens,
        policy_decisions=(decision_b,),
    )
    hash_a = compute_context_decision_evidence_fingerprint(assembly_a)
    hash_b = compute_context_decision_evidence_fingerprint(assembly_b)
    assert hash_a != hash_b
    assert compute_model_facing_messages_hash(assembly_a.messages) == compute_model_facing_messages_hash(
        assembly_b.messages
    )


def test_txp4r1_q10_fingerprint_excludes_raw_content() -> None:
    secret = "super-secret-fragment-body"
    from intergrax.context.contracts import ContextFragment, ContextFragmentSource

    fragment = ContextFragment(
        fragment_id="f1",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_id="s1",
        content=secret,
        token_estimate=1,
        relevance_score=0.5,
        freshness_score=0.5,
        confidence_score=0.5,
        mandatory=False,
    )
    assembled = AssembledContext(
        messages=(ChatMessage(role="user", content="visible"),),
        fragments_included=(fragment,),
        fragments_excluded=(),
        provenance=(),
        total_tokens=1,
        budget_tokens=100,
    )
    canonical = build_context_decision_evidence_canonical(assembled)
    blob = str(canonical)
    assert secret not in blob


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
        policy_decisions=(
            ContextPolicyDecision(
                stage=ContextPolicyStage.EXACT_DEDUP,
                strategy_id="dedup-default",
                input_fragment_ids=(),
                output_fragment_ids=(),
                reason_code=ContextPolicyReasonCode.EXACT_DUPLICATE_CONTENT,
            ),
        ),
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
    ctx_payload = parse_context_assembly_payload(ctx_events[0])
    llm_payload = parse_llm_call_payload(llm_events[0])
    assert ctx_payload is not None and llm_payload is not None
    assert ctx_payload.model_input_messages_hash == expected_hash
    assert llm_payload.model_input_messages_hash == expected_hash
    verdict = attribute_model_call_to_context(ctx_events[0], llm_events[0])
    assert verdict.attributable, verdict.reason
    assert verdict.evidence is not None
    assert verdict.evidence.context_decision_evidence_fingerprint == recompute_context_decision_fingerprint(
        assembled
    )
    assert isinstance(llm_payload, LlmCallPayloadV3)
    assert llm_payload.execution_scope == ModelCallExecutionScope.PRIMARY_MODEL_CALL
    assert llm_payload.context_assembly_event_id == str(ctx_events[0].event_id)
    assert llm_payload.provider == "openai"


def test_txp4r1_q15_same_execution_two_assemblies_distinguishable(governed_execution) -> None:
    run_id, task_id, _attempt_id, _execution_id, bus = governed_execution
    messages = (ChatMessage(role="user", content="repeat"),)
    for suffix in ("a", "b"):
        assembled = AssembledContext(
            messages=messages,
            fragments_included=(),
            fragments_excluded=(),
            provenance=(),
            total_tokens=1,
            budget_tokens=100,
            compaction_strategy_id=f"compaction-{suffix}",
        )
        record_context_assembled_from_engine(
            bus,
            assembled=assembled,
            task_id=task_id,
            run_id=run_id,
        )
        _record_primary_model_call(bus, task_id=task_id, run_id=run_id, messages=messages)

    ctx_events = [e for e in bus.history if e.event_type == RuntimeEventType.CONTEXT_ASSEMBLED]
    llm_events = [e for e in bus.history if e.event_type == RuntimeEventType.LLM_CALL]
    assert len(ctx_events) == 2 and len(llm_events) == 2
    assert ctx_events[0].event_id != ctx_events[1].event_id
    v0 = attribute_model_call_to_context(ctx_events[0], llm_events[0])
    v1 = attribute_model_call_to_context(ctx_events[1], llm_events[1])
    assert v0.attributable and v1.attributable
    assert attribute_model_call_to_context(ctx_events[0], llm_events[1]).attributable is False


def test_txp4_q13_same_run_multi_execution_isolation(governed_execution) -> None:
    run_id, task_id, attempt_id, _execution_id, bus = governed_execution
    messages_a = (ChatMessage(role="user", content="a"),)
    messages_b = (ChatMessage(role="user", content="b"),)

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
    payload = parse_llm_call_payload(llm)
    assert isinstance(payload, LlmCallPayloadV3)
    assert payload.execution_scope == ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL


def test_txp4r1_q20_cross_tenant_attribution_rejected(governed_execution) -> None:
    run_id, task_id, _attempt_id, _execution_id, bus = governed_execution
    messages = (ChatMessage(role="user", content="x"),)
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
    )
    ctx = next(e for e in bus.history if e.event_type == RuntimeEventType.CONTEXT_ASSEMBLED)
    ctx_tenant = ctx.model_copy(update={"tenant_id": "tenant-a"})
    _record_primary_model_call(
        bus,
        task_id=task_id,
        run_id=run_id,
        messages=messages,
        tenant_id="tenant-b",
    )
    llm = next(e for e in reversed(bus.history) if e.event_type == RuntimeEventType.LLM_CALL)
    verdict = attribute_model_call_to_context(ctx_tenant, llm)
    assert verdict.attributable is False
    assert verdict.reason == "tenant_mismatch"


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


def test_txp4r1_q25_streaming_not_production_primary() -> None:
    result = subprocess.run(
        [
            "git",
            "grep",
            "-E",
            r"\.stream_messages\(|\.stream_with_tools\(",
            "--",
            "agents",
            "intergrax/runtime/nexus",
            "intergrax/applications",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert result.stdout.strip() == ""
