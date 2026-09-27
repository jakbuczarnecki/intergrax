# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default qualified Tool invocation resolver — mapping only (S24-GAP-02-P3)."""

from __future__ import annotations

from pydantic import BaseModel

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.execution_identity import TaskId


class DefaultQualifiedToolInvocationResolver:
    """Construct canonical ExecutionBoundCatalogToolInvokeRequest deterministically."""

    def resolve(
        self,
        *,
        activated_tool_id: str,
        selected_operation: str,
        material: BaseModel,
        tenant_id: str,
        task_id: TaskId,
        run_id: str,
        agent_id: str,
        step_id: str,
        execution_request_id: str,
        correlation_request_id: str | None,
        idempotency_key: str | None,
    ) -> ExecutionBoundCatalogToolInvokeRequest:
        # Atomic runtime callable is selected by activated_tool_id; selected_operation
        # remains semantic execution intent, not a ToolRuntime sub-operation selector.
        cleaned_operation = require_non_empty_text(
            selected_operation,
            label="selected_operation",
        )
        cleaned_tool = require_non_empty_text(activated_tool_id, label="activated_tool_id")
        cleaned_tenant = require_non_empty_text(tenant_id, label="tenant_id")
        cleaned_run = require_non_empty_text(run_id, label="run_id")
        cleaned_agent = require_non_empty_text(agent_id, label="agent_id")
        cleaned_step = require_non_empty_text(step_id, label="step_id")
        cleaned_execution_request = require_non_empty_text(
            execution_request_id,
            label="execution_request_id",
        )
        if not isinstance(material, BaseModel):
            raise TypeError("material must be BaseModel")
        resolved_idempotency = idempotency_key or (
            f"qmte:{cleaned_execution_request}:{cleaned_operation}"
        )
        return ExecutionBoundCatalogToolInvokeRequest(
            tool_id=cleaned_tool,
            input=material,
            tenant_id=cleaned_tenant,
            task_id=str(task_id),
            run_id=cleaned_run,
            agent_id=cleaned_agent,
            step_id=cleaned_step,
            correlation_request_id=correlation_request_id,
            idempotency_key=resolved_idempotency,
        )


__all__ = ["DefaultQualifiedToolInvocationResolver"]
