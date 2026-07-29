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
    FailurePhase,
    SafeErrorCode,
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
    failure_phase: str | None = None
    error_type: str | None = None
    safe_error_code: str | None = None


def build_protocol_messages(
    request: ConversationPlanningRequest,
    protocol: str,
) -> list[ChatMessage]:
    messages = list(build_planning_messages(request))
    if protocol == PROTOCOL_SINGLE_PLAN_TOOL:
        transport = ChatMessage(role="system", content=_TOOL_TRANSPORT_INSTRUCTION)
        messages.insert(1, transport)
    return messages


def _failure(
    *,
    category: str,
    phase: FailurePhase,
    error_type: str | None = None,
    safe_error_code: str | None = None,
) -> ProtocolAttemptSuccess:
    return ProtocolAttemptSuccess(
        ok=False,
        failure_category=category,
        failure_phase=phase.value,
        error_type=error_type,
        safe_error_code=safe_error_code,
    )


def _is_resource_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if any(token in name for token in ("memory", "oom", "cuda")):
        return True
    return any(
        token in message
        for token in ("out of memory", "cuda oom", "resource exhausted", "insufficient memory")
    )


def _classify_provider_error(exc: BaseException, *, protocol: str) -> tuple[str, SafeErrorCode]:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if _is_resource_error(exc):
        return StructuralFailureCategory.RESOURCE_LIMIT.value, SafeErrorCode.OLLAMA_RESOURCE_LIMIT
    if protocol == PROTOCOL_SINGLE_PLAN_TOOL:
        if "tool" in message and "choice" in message:
            return StructuralFailureCategory.PROVIDER_ERROR.value, SafeErrorCode.OLLAMA_TOOL_CHOICE_REJECTED
        if "tool" in message and "schema" in message:
            return StructuralFailureCategory.PROVIDER_ERROR.value, SafeErrorCode.OLLAMA_TOOL_SCHEMA_REJECTED
        if "tool" in message and ("unsupported" in message or "not support" in message):
            return StructuralFailureCategory.PROVIDER_ERROR.value, SafeErrorCode.OLLAMA_MODEL_TOOLS_UNSUPPORTED
        if "transport" in message or "connection" in message or "timeout" in message:
            return (
                StructuralFailureCategory.PROVIDER_ERROR.value,
                SafeErrorCode.OLLAMA_PROVIDER_TRANSPORT_FAILED,
            )
        if "invalid" in message and "response" in message:
            return (
                StructuralFailureCategory.PROVIDER_ERROR.value,
                SafeErrorCode.OLLAMA_PROVIDER_RESPONSE_INVALID,
            )
    if "timeout" in name or "connection" in name:
        return StructuralFailureCategory.PROVIDER_ERROR.value, SafeErrorCode.OLLAMA_PROVIDER_TRANSPORT_FAILED
    return StructuralFailureCategory.PROVIDER_ERROR.value, SafeErrorCode.UNKNOWN_PROVIDER_FAILURE


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
                return _failure(
                    category=StructuralFailureCategory.PROTOCOL_UNSUPPORTED.value,
                    phase=FailurePhase.CAPABILITY_CHECK,
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
                return _failure(
                    category=StructuralFailureCategory.DRAFT_VALIDATION_FAILED.value,
                    phase=FailurePhase.DRAFT_VALIDATION,
                    error_type=type(parsed).__name__,
                )
            draft = parsed
        elif protocol == PROTOCOL_SINGLE_PLAN_TOOL:
            if not adapter.supports_tools():
                return _failure(
                    category=StructuralFailureCategory.PROTOCOL_UNSUPPORTED.value,
                    phase=FailurePhase.CAPABILITY_CHECK,
                    safe_error_code=SafeErrorCode.OLLAMA_MODEL_TOOLS_UNSUPPORTED.value,
                )
            result = adapter.generate_with_tools(
                messages,
                [SUBMIT_DRAFT_TOOL_SCHEMA],
                temperature=benchmark.temperature,
                max_tokens=benchmark.max_tokens,
                tool_choice="auto",
                run_id=run_id,
            )
            tool_calls = result.tool_calls
            if not tool_calls:
                return _failure(
                    category=StructuralFailureCategory.MISSING_PLAN_TOOL_CALL.value,
                    phase=FailurePhase.TOOL_CALL_VALIDATION,
                )
            if len(tool_calls) != 1:
                return _failure(
                    category=StructuralFailureCategory.MULTIPLE_PLAN_TOOL_CALLS.value,
                    phase=FailurePhase.TOOL_CALL_VALIDATION,
                )
            tool_call = tool_calls[0]
            if tool_call.name != SUBMIT_DRAFT_TOOL_NAME:
                return _failure(
                    category=StructuralFailureCategory.UNEXPECTED_PLAN_TOOL.value,
                    phase=FailurePhase.TOOL_CALL_VALIDATION,
                )
            try:
                json.loads(tool_call.arguments_json)
            except json.JSONDecodeError:
                return _failure(
                    category=StructuralFailureCategory.INVALID_TOOL_ARGUMENTS.value,
                    phase=FailurePhase.TOOL_CALL_VALIDATION,
                )
            try:
                draft = ConversationInteractionDraft.model_validate_json(tool_call.arguments_json)
            except ValidationError:
                return _failure(
                    category=StructuralFailureCategory.DRAFT_VALIDATION_FAILED.value,
                    phase=FailurePhase.DRAFT_VALIDATION,
                    error_type="ValidationError",
                )
        else:
            return _failure(
                category=StructuralFailureCategory.PROTOCOL_UNSUPPORTED.value,
                phase=FailurePhase.CAPABILITY_CHECK,
            )

        try:
            plan = compile_interaction_draft(draft, request)
        except ConversationDraftCompilationError:
            return _failure(
                category=StructuralFailureCategory.DRAFT_COMPILATION_FAILED.value,
                phase=FailurePhase.DRAFT_COMPILATION,
                error_type="ConversationDraftCompilationError",
            )
        try:
            validate_plan_against_request(plan, request)
        except PlanRequestValidationError:
            return _failure(
                category=StructuralFailureCategory.CANONICAL_VALIDATION_FAILED.value,
                phase=FailurePhase.CANONICAL_VALIDATION,
                error_type="PlanRequestValidationError",
            )
        return ProtocolAttemptSuccess(ok=True, draft=draft, plan=plan)
    except ValidationError:
        return _failure(
            category=StructuralFailureCategory.DRAFT_VALIDATION_FAILED.value,
            phase=FailurePhase.DRAFT_VALIDATION,
            error_type="ValidationError",
        )
    except Exception as exc:
        category, safe_code = _classify_provider_error(exc, protocol=protocol)
        phase = FailurePhase.PROVIDER_INVOKE
        if category == StructuralFailureCategory.RESOURCE_LIMIT.value:
            phase = FailurePhase.PROVIDER_INVOKE
        return _failure(
            category=category,
            phase=phase,
            error_type=type(exc).__name__,
            safe_error_code=safe_code.value,
        )
