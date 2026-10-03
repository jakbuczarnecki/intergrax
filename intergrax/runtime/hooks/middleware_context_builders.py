# © Artur Czarnecki. All rights reserved.

"""Construct typed middleware fields on :class:`HookContext` for canonical pipeline producers."""

from __future__ import annotations

from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.contracts.middleware_hook_semantics import (
    ContextBuildHookPayload,
    DataProtectionHookPayload,
    LlmInferenceHookPayload,
    MiddlewareExecutionSubjectFacet,
    TaskIntakeHookPayload,
    ToolCallHookPayload,
)
from intergrax.runtime.middleware.hook_semantic_adapters import (
    data_protection_from_memory_write_state,
)


def stringify_argument_map(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def tool_call_payload_from_runtime_state(
    runtime_state: dict[str, object],
) -> ToolCallHookPayload | None:
    tool_id = runtime_state.get("tool_id") or runtime_state.get("tool_name")
    if not tool_id:
        return None
    autonomy_raw = runtime_state.get("autonomy_level")
    autonomy = (
        AutonomyLevel(str(autonomy_raw)) if autonomy_raw is not None else None
    )
    risk_raw = runtime_state.get("agent_risk_level")
    risk = (
        AgentRiskLevel(str(risk_raw)) if risk_raw is not None else None
    )
    return ToolCallHookPayload(
        tool_id=str(tool_id),
        tool_name=_optional_str(runtime_state.get("tool_name")),
        request_id=_optional_str(runtime_state.get("request_id")),
        arguments=stringify_argument_map(runtime_state.get("arguments")),
        capability_ids=_string_list(runtime_state.get("capability_ids")),
        allowed_tool_ids=_string_list(runtime_state.get("allowed_tool_ids")),
        autonomy_level=autonomy,
        agent_risk_level=risk,
    )


def llm_payload_from_runtime_state(runtime_state: dict[str, object]) -> LlmInferenceHookPayload:
    prompt = _optional_str(runtime_state.get("prompt"))
    llm_output = _optional_str(
        runtime_state.get("llm_output") or runtime_state.get("output"),
    )
    return LlmInferenceHookPayload(prompt=prompt, llm_output=llm_output)


def task_intake_payload_from_runtime_state(
    runtime_state: dict[str, object],
) -> TaskIntakeHookPayload:
    return TaskIntakeHookPayload(
        capability=_optional_str(runtime_state.get("capability")),
        classification=_optional_str(runtime_state.get("classification")),
    )


def context_build_payload_from_runtime_state(
    runtime_state: dict[str, object],
) -> ContextBuildHookPayload:
    return ContextBuildHookPayload(
        message=_optional_str(runtime_state.get("message")),
        capability=_optional_str(runtime_state.get("capability")),
    )


def subject_from_runtime_state(runtime_state: dict[str, object]) -> MiddlewareExecutionSubjectFacet:
    return MiddlewareExecutionSubjectFacet(
        tenant_id=_optional_str(runtime_state.get("tenant_id")),
        resource_tenant_id=_optional_str(runtime_state.get("resource_tenant_id")),
        user_id=_optional_str(runtime_state.get("user_id")),
    )


def sync_typed_fields_from_runtime_state(
    runtime_state: dict[str, object],
    *,
    prefer_data_protection: bool = False,
) -> tuple[MiddlewareExecutionSubjectFacet, object]:
    subject = subject_from_runtime_state(runtime_state)
    if prefer_data_protection or "memory_write" in runtime_state or "value" in runtime_state:
        payload: object = data_protection_from_memory_write_state(runtime_state)
        return subject, payload
    if runtime_state.get("tool_id") or runtime_state.get("tool_name"):
        tool_payload = tool_call_payload_from_runtime_state(runtime_state)
        if tool_payload is not None:
            return subject, tool_payload
    if runtime_state.get("prompt") or runtime_state.get("llm_output") or runtime_state.get("output"):
        return subject, llm_payload_from_runtime_state(runtime_state)
    if runtime_state.get("capability") or runtime_state.get("classification"):
        return subject, task_intake_payload_from_runtime_state(runtime_state)
    if runtime_state.get("message"):
        return subject, context_build_payload_from_runtime_state(runtime_state)
    from intergrax.contracts.middleware_hook_semantics import EmptyMiddlewareHookPayload

    return subject, EmptyMiddlewareHookPayload()


def _optional_str(raw: object) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _string_list(raw: object) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item) for item in raw]
