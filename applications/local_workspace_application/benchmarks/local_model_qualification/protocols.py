# © Artur Czarnecki. All rights reserved.

"""Protocol adapters for structured output and single-plan-tool submission."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, Sequence

from pydantic import ValidationError

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

from local_workspace_application.benchmarks.local_model_qualification.config import BenchmarkConfig
from local_workspace_application.benchmarks.local_model_qualification.contracts import (
    StructuralFailureCategory,
)
from local_workspace_application.conversation.interaction_draft_models import ConversationInteractionDraft
from local_workspace_application.conversation.interaction_models import ConversationPlanningRequest
from local_workspace_application.conversation.interaction_plan_compiler import (
    ConversationDraftCompilationError,
    compile_interaction_draft,
)
from local_workspace_application.conversation.interaction_planner import PlanRequestValidationError, validate_plan_against_request
from local_workspace_application.conversation.interaction_prompt import build_planning_messages

PROTOCOL_STRUCTURED_OUTPUT = "structured_output"
PROTOCOL_SINGLE_PLAN_TOOL = "single_plan_tool"

SUBMIT_DRAFT_TOOL_NAME = "submit_conversation_interaction_draft"

_TOOL_TRANSPORT_INSTRUCTION = (
    "Call submit_conversation_interaction_draft exactly once with the complete "
    "semantic draft as its arguments. This submission tool does not execute any "
    "operation. Do not answer in plain text and do not call any other tool."
)

SUBMIT_DRAFT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": SUBMIT_DRAFT_TOOL_NAME,
        "description": (
            "Submit one complete LKW semantic interaction draft for deterministic "
            "compilation and validation. This tool does not execute actions."
        ),
        "parameters": ConversationInteractionDraft.model_json_schema(),
    },
}


class BenchmarkAdapter(Protocol):
    def supports_structured_output(self) -> bool: ...

    def supports_tools(self) -> bool: ...

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ): ...

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: str | dict | None = None,
        run_id: str | None = None,
    ): ...


@dataclass(frozen=True, slots=True)
class ProtocolAttemptSuccess:
    ok: bool = True
    draft: ConversationInteractionDraft | None = None
    plan: object | None = None
    failure_category: str | None = None
    error_type: str | None = None


def build_protocol_messages(
    request: ConversationPlanningRequest,
    protocol: str,
) -> list[ChatMessage]:
    messages = list(build_planning_messages(request))
    if protocol == PROTOCOL_SINGLE_PLAN_TOOL:
        transport = ChatMessage(role="system", content=_TOOL_TRANSPORT_INSTRUCTION)
        messages.insert(1, transport)
    return messages


def _is_resource_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    return any(token in name for token in ("memory", "resource", "oom", "cuda"))


def run_protocol_attempt(
    *,
    adapter: BenchmarkAdapter,
    protocol: str,
    request: ConversationPlanningRequest,
    benchmark: BenchmarkConfig,
    run_id: str,
) -> ProtocolAttemptSuccess:
    messages = build_protocol_messages(request, protocol)
    draft: ConversationInteractionDraft | None = None
    try:
        if protocol == PROTOCOL_STRUCTURED_OUTPUT:
            if not adapter.supports_structured_output():
                return ProtocolAttemptSuccess(
                    ok=False,
                    failure_category=StructuralFailureCategory.PROTOCOL_UNSUPPORTED.value,
                )
            result = adapter.generate_structured(
                messages,
                ConversationInteractionDraft,
                temperature=benchmark.temperature,
                max_tokens=benchmark.max_tokens,
                run_id=run_id,
            )
            parsed = result.parsed
            if not isinstance(parsed, ConversationInteractionDraft):
                return ProtocolAttemptSuccess(
                    ok=False,
                    failure_category=StructuralFailureCategory.DRAFT_VALIDATION_FAILED.value,
                    error_type=type(parsed).__name__,
                )
            draft = parsed
        elif protocol == PROTOCOL_SINGLE_PLAN_TOOL:
            if not adapter.supports_tools():
                return ProtocolAttemptSuccess(
                    ok=False,
                    failure_category=StructuralFailureCategory.PROTOCOL_UNSUPPORTED.value,
                )
            result = adapter.generate_with_tools(
                messages,
                [SUBMIT_DRAFT_TOOL_SCHEMA],
                temperature=benchmark.temperature,
                max_tokens=benchmark.max_tokens,
                tool_choice="required",
                run_id=run_id,
            )
            tool_calls = result.tool_calls
            if not tool_calls:
                return ProtocolAttemptSuccess(
                    ok=False,
                    failure_category=StructuralFailureCategory.MISSING_PLAN_TOOL_CALL.value,
                )
            if len(tool_calls) != 1:
                return ProtocolAttemptSuccess(
                    ok=False,
                    failure_category=StructuralFailureCategory.MULTIPLE_PLAN_TOOL_CALLS.value,
                )
            tool_call = tool_calls[0]
            if tool_call.name != SUBMIT_DRAFT_TOOL_NAME:
                return ProtocolAttemptSuccess(
                    ok=False,
                    failure_category=StructuralFailureCategory.UNEXPECTED_PLAN_TOOL.value,
                )
            try:
                json.loads(tool_call.arguments_json)
            except json.JSONDecodeError:
                return ProtocolAttemptSuccess(
                    ok=False,
                    failure_category=StructuralFailureCategory.INVALID_TOOL_ARGUMENTS.value,
                )
            try:
                draft = ConversationInteractionDraft.model_validate_json(tool_call.arguments_json)
            except ValidationError:
                return ProtocolAttemptSuccess(
                    ok=False,
                    failure_category=StructuralFailureCategory.DRAFT_VALIDATION_FAILED.value,
                    error_type="ValidationError",
                )
        else:
            return ProtocolAttemptSuccess(
                ok=False,
                failure_category=StructuralFailureCategory.PROTOCOL_UNSUPPORTED.value,
            )

        plan = compile_interaction_draft(draft, request)
        validate_plan_against_request(plan, request)
        return ProtocolAttemptSuccess(ok=True, draft=draft, plan=plan)
    except ConversationDraftCompilationError:
        return ProtocolAttemptSuccess(
            ok=False,
            failure_category=StructuralFailureCategory.DRAFT_COMPILATION_FAILED.value,
            error_type="ConversationDraftCompilationError",
        )
    except PlanRequestValidationError:
        return ProtocolAttemptSuccess(
            ok=False,
            failure_category=StructuralFailureCategory.CANONICAL_VALIDATION_FAILED.value,
            error_type="PlanRequestValidationError",
        )
    except ValidationError:
        return ProtocolAttemptSuccess(
            ok=False,
            failure_category=StructuralFailureCategory.DRAFT_VALIDATION_FAILED.value,
            error_type="ValidationError",
        )
    except Exception as exc:
        if _is_resource_error(exc):
            return ProtocolAttemptSuccess(
                ok=False,
                failure_category=StructuralFailureCategory.RESOURCE_LIMIT.value,
                error_type=type(exc).__name__,
            )
        return ProtocolAttemptSuccess(
            ok=False,
            failure_category=StructuralFailureCategory.PROVIDER_ERROR.value,
            error_type=type(exc).__name__,
        )
