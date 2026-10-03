# © Artur Czarnecki. All rights reserved.

"""Test helpers for typed middleware HookContext construction."""

from __future__ import annotations

from intergrax.contracts.data_classification import DataClassification
from intergrax.contracts.middleware_hook_semantics import (
    DataProtectionHookPayload,
    DataProtectionRestrictedValue,
    LlmInferenceHookPayload,
    MiddlewareExecutionSubjectFacet,
    ToolCallHookPayload,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.hooks.hook_context import HookContext


def tool_hook_context_for_test(
    *,
    tool_id: str,
    arguments: dict[str, str] | None = None,
    **kwargs: object,
) -> HookContext:
    runtime_state = {
        "tool_id": tool_id,
        "arguments": arguments or {},
    }
    return HookContext(
        task_id=str(kwargs.get("task_id", "task-1")),
        run_id=str(kwargs.get("run_id", "run-1")),
        agent_id=kwargs.get("agent_id"),  # type: ignore[arg-type]
        phase=kwargs.get("phase") or ExecutionPhase.STEP_EXECUTION,
        payload=ToolCallHookPayload(
            tool_id=tool_id,
            arguments=arguments or {},
        ),
        runtime_state=runtime_state,
    )


def llm_hook_context_for_test(
    *,
    prompt: str,
    **kwargs: object,
) -> HookContext:
    return HookContext(
        task_id=str(kwargs.get("task_id", "run-1")),
        run_id=str(kwargs.get("run_id", "run-1")),
        phase=kwargs.get("phase") or ExecutionPhase.STEP_EXECUTION,
        payload=LlmInferenceHookPayload(prompt=prompt),
        runtime_state={"prompt": prompt},
    )


def tenant_intake_hook_context_for_test(
    *,
    tenant_id: str,
    resource_tenant_id: str | None = None,
    user_id: str | None = None,
    **kwargs: object,
) -> HookContext:
    runtime_state = {
        "tenant_id": tenant_id,
        "resource_tenant_id": resource_tenant_id or tenant_id,
        "user_id": user_id or "user-1",
    }
    return HookContext(
        task_id=str(kwargs.get("task_id", "run-1")),
        run_id=str(kwargs.get("run_id", "run-1")),
        agent_id=kwargs.get("agent_id"),  # type: ignore[arg-type]
        phase=kwargs.get("phase") or ExecutionPhase.INTAKE,
        subject=MiddlewareExecutionSubjectFacet(
            tenant_id=tenant_id,
            resource_tenant_id=resource_tenant_id or tenant_id,
            user_id=user_id or "user-1",
        ),
        runtime_state=runtime_state,
    )


def data_protection_hook_context_for_test(
    *,
    classification: DataClassification = DataClassification.RESTRICTED,
    secret: str | None = "x",
    note: str | None = None,
    **kwargs: object,
) -> HookContext:
    value_model = DataProtectionRestrictedValue(
        data_classification=classification,
        secret=secret,
    )
    runtime_value: dict[str, str] = {"data_classification": classification.value}
    if secret is not None:
        runtime_value["secret"] = secret
    if note is not None:
        runtime_value["note"] = note
    payload = DataProtectionHookPayload(value=value_model)
    return HookContext(
        task_id=str(kwargs.get("task_id", "task-1")),
        run_id=str(kwargs.get("run_id", "run-1")),
        agent_id=kwargs.get("agent_id"),  # type: ignore[arg-type]
        payload=payload,
        runtime_state={"value": runtime_value},
    )
