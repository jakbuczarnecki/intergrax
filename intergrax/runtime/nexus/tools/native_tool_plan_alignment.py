# © Artur Czarnecki. All rights reserved.

"""ENG-6 — validate native LLM tool calls against the executable ToolCallPlan batch."""

from __future__ import annotations

import json
from collections.abc import Sequence

from intergrax.llm_adapters.contracts.tool_call import LLMToolCall, validate_tool_call_identities
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan


class NativeToolPlanAlignmentError(ValueError):
    """Native LLM tool calls do not match the planned executable tool batch."""


def _canonical_llm_arguments(arguments_json: str) -> dict[str, object]:
    try:
        parsed = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as exc:
        raise NativeToolPlanAlignmentError(
            "native tool call arguments JSON is malformed"
        ) from exc
    if not isinstance(parsed, dict):
        raise NativeToolPlanAlignmentError(
            "native tool call arguments must be a JSON object"
        )
    return parsed


def validate_native_tool_plan_alignment(
    llm_tool_calls: Sequence[LLMToolCall],
    tool_plan: ToolCallPlan,
) -> None:
    """Ensure provider-visible native calls match the ToolCallPlan execution batch."""
    validate_tool_call_identities(llm_tool_calls)
    planned_calls = tool_plan.calls
    if len(llm_tool_calls) != len(planned_calls):
        raise NativeToolPlanAlignmentError(
            "native tool call count does not match planned tool call count"
        )

    for index, (llm_call, planned_call) in enumerate(
        zip(llm_tool_calls, planned_calls, strict=True)
    ):
        if llm_call.name != planned_call.tool_id:
            raise NativeToolPlanAlignmentError(
                f"native tool call name mismatch at index {index}: "
                f"{llm_call.name!r} != {planned_call.tool_id!r}"
            )
        llm_args = _canonical_llm_arguments(llm_call.arguments_json)
        try:
            validated = type(planned_call.input).model_validate(llm_args)
        except Exception as exc:
            raise NativeToolPlanAlignmentError(
                f"native tool call arguments invalid at index {index}"
            ) from exc
        if validated.model_dump() != planned_call.input.model_dump():
            raise NativeToolPlanAlignmentError(
                f"native tool call arguments mismatch at index {index}"
            )
