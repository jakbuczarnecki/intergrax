# © Artur Czarnecki. All rights reserved.

"""DS-E2E-12 — deterministic tests for typed planner action-context transport."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.tools.investigation_proof import (
    build_completed_observation_reference_index,
)
from intergrax.runtime.nexus.tools.native_planner_action_context import (
    PLANNER_ACTION_CONTEXT_TOOL_ID,
    NativePlannerActionContextError,
    append_planner_action_context_schema,
    parse_planner_action_context_call,
    process_native_planner_tool_response,
    split_native_planner_tool_calls,
)
from intergrax.runtime.nexus.tools.native_tool_plan_alignment import (
    validate_native_tool_plan_alignment,
)
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _ProbeIn(BaseModel):
    service: str = "checkout"


def _annotation_call(
    *,
    call_id: str = "ann-1",
    basis: list[str] | None = None,
    purpose: str = "Correlate checkout errors with upstream dependency.",
    extra_field: str | None = None,
) -> LLMToolCall:
    payload: dict[str, object] = {
        "evidence_basis_references": basis if basis is not None else ["obs.ref.a"],
        "purpose": purpose,
    }
    if extra_field is not None:
        payload["hypothesis"] = extra_field
    return LLMToolCall(
        id=call_id,
        name=PLANNER_ACTION_CONTEXT_TOOL_ID,
        arguments_json=json.dumps(payload),
    )


def _business_call(
    *,
    call_id: str = "biz-1",
    tool_name: str = "probe.fetch_logs",
) -> LLMToolCall:
    return LLMToolCall(
        id=call_id,
        name=tool_name,
        arguments_json=json.dumps({"service": "checkout"}),
    )


def _response(*calls: LLMToolCall) -> LLMAdapterResponse:
    return LLMAdapterResponse(
        content="",
        tool_calls=tuple(calls),
        response_id="resp-atomic-1",
    )


def _reference_index_with_prior() -> dict[str, str]:
    from intergrax.llm.messages import ChatMessage

    messages = [
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_prior",
                    "type": "function",
                    "function": {"name": "probe.a", "arguments": "{}"},
                }
            ],
        ),
        ChatMessage(
            role="tool",
            content=json.dumps({"evidence_reference": "obs.ref.a", "payload": {}}),
            tool_call_id="call_prior",
        ),
    ]
    return build_completed_observation_reference_index(messages)


def test_split_annotation_and_business_calls() -> None:
    split = split_native_planner_tool_calls(
        (_annotation_call(), _business_call(), _business_call(call_id="biz-2"))
    )
    assert len(split.annotation_calls) == 1
    assert len(split.business_tool_calls) == 2
    assert split.annotation_calls[0].name == PLANNER_ACTION_CONTEXT_TOOL_ID


def test_process_atomic_annotation_plus_business_success() -> None:
    reference_index = _reference_index_with_prior()
    processed = process_native_planner_tool_response(
        _response(_annotation_call(basis=["obs.ref.a"]), _business_call()),
        available_evidence_references=frozenset(reference_index),
        reference_index=reference_index,
    )
    assert processed.is_executable_investigation_round is True
    assert processed.transport.action_context is not None
    assert processed.transport.action_context.purpose
    assert processed.transport.business_tool_calls[0].name == "probe.fetch_logs"


def test_missing_annotation_with_prior_evidence_fails_closed() -> None:
    reference_index = _reference_index_with_prior()
    with pytest.raises(NativePlannerActionContextError, match="exactly one"):
        process_native_planner_tool_response(
            _response(_business_call()),
            available_evidence_references=frozenset(reference_index),
            reference_index=reference_index,
        )


def test_empty_basis_with_prior_evidence_fails_closed() -> None:
    reference_index = _reference_index_with_prior()
    with pytest.raises(NativePlannerActionContextError, match="explicit evidence basis"):
        process_native_planner_tool_response(
            _response(_annotation_call(basis=[]), _business_call()),
            available_evidence_references=frozenset(reference_index),
            reference_index=reference_index,
        )


def test_unknown_basis_fails_closed() -> None:
    reference_index = _reference_index_with_prior()
    with pytest.raises(NativePlannerActionContextError, match="unknown basis"):
        process_native_planner_tool_response(
            _response(_annotation_call(basis=["obs.unknown"]), _business_call()),
            available_evidence_references=frozenset(reference_index),
            reference_index=reference_index,
        )


def test_duplicate_basis_fails_closed() -> None:
    reference_index = _reference_index_with_prior()
    with pytest.raises(NativePlannerActionContextError, match="duplicate basis"):
        parse_planner_action_context_call(
            _annotation_call(basis=["obs.ref.a", "obs.ref.a"])
        )


def test_extra_annotation_field_rejected() -> None:
    with pytest.raises(NativePlannerActionContextError, match="schema validation failed"):
        parse_planner_action_context_call(
            _annotation_call(extra_field="hidden reasoning")
        )


def test_multiple_annotations_fail_closed() -> None:
    reference_index = _reference_index_with_prior()
    with pytest.raises(NativePlannerActionContextError, match="at most one annotation"):
        process_native_planner_tool_response(
            _response(
                _annotation_call(call_id="ann-1"),
                _annotation_call(call_id="ann-2"),
                _business_call(),
            ),
            available_evidence_references=frozenset(reference_index),
            reference_index=reference_index,
        )


def test_annotation_only_is_not_executable_round() -> None:
    reference_index = _reference_index_with_prior()
    processed = process_native_planner_tool_response(
        _response(_annotation_call(basis=["obs.ref.a"])),
        available_evidence_references=frozenset(reference_index),
        reference_index=reference_index,
    )
    assert processed.is_executable_investigation_round is False
    assert processed.transport.business_tool_calls == ()


def test_first_round_annotation_optional() -> None:
    processed = process_native_planner_tool_response(
        _response(_business_call()),
        available_evidence_references=frozenset(),
        reference_index={},
    )
    assert processed.is_executable_investigation_round is True
    assert processed.transport.action_context is None


def test_multiple_business_calls_share_one_annotation() -> None:
    reference_index = _reference_index_with_prior()
    processed = process_native_planner_tool_response(
        _response(
            _annotation_call(basis=["obs.ref.a"]),
            _business_call(call_id="biz-1"),
            _business_call(call_id="biz-2", tool_name="probe.metrics"),
        ),
        available_evidence_references=frozenset(reference_index),
        reference_index=reference_index,
    )
    assert processed.transport.action_context is not None
    assert len(processed.transport.business_tool_calls) == 2
    next_ids = tuple(call.id for call in processed.transport.business_tool_calls)
    assert next_ids == ("biz-1", "biz-2")


def test_annotation_not_in_business_plan_alignment() -> None:
    reference_index = _reference_index_with_prior()
    processed = process_native_planner_tool_response(
        _response(_annotation_call(basis=["obs.ref.a"]), _business_call(call_id="biz-1")),
        available_evidence_references=frozenset(reference_index),
        reference_index=reference_index,
    )
    business_calls = processed.transport.business_tool_calls
    tool_plan = ToolCallPlan(
        calls=[
            PlannedToolCall(
                step_id="tool",
                tool_id="probe.fetch_logs",
                input=_ProbeIn(),
            )
        ]
    )
    next_tool_call_ids = tuple(call.id for call in business_calls)
    tool_plan_ids = tuple(call.tool_id for call in tool_plan.calls)
    assert PLANNER_ACTION_CONTEXT_TOOL_ID not in next_tool_call_ids
    assert PLANNER_ACTION_CONTEXT_TOOL_ID not in tool_plan_ids
    validate_native_tool_plan_alignment(business_calls, tool_plan)


def test_append_schema_does_not_mutate_business_catalog() -> None:
    business = [
        {
            "type": "function",
            "function": {
                "name": "probe.fetch_logs",
                "description": "fetch",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    combined = append_planner_action_context_schema(business)
    assert len(business) == 1
    assert len(combined) == 2
    assert combined[-1]["function"]["name"] == PLANNER_ACTION_CONTEXT_TOOL_ID


def test_budget_annotation_excluded_from_business_call_count() -> None:
    response = _response(_annotation_call(), _business_call())
    split = split_native_planner_tool_calls(response.tool_calls)
    assert len(response.tool_calls) == 2
    assert len(split.business_tool_calls) == 1
