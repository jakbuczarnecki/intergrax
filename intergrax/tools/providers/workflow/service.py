# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Workflow orchestrator catalog tool services."""

from __future__ import annotations

from intergrax.integrations.contracts.workflow_orchestrator import WorkflowOrchestratorBackend
from intergrax.tools.providers.workflow.contracts import (
    WorkflowFetchLogsInput,
    WorkflowFetchLogsOutput,
    WorkflowPollInput,
    WorkflowPollOutput,
    WorkflowTriggerInput,
    WorkflowTriggerOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

WORKFLOW_TRIGGER_TOOL_ID = "workflow.trigger"
WORKFLOW_POLL_TOOL_ID = "workflow.poll"
WORKFLOW_FETCH_LOGS_TOOL_ID = "workflow.fetch_logs"


def _require_orchestrator(ctx: ToolWiringContext) -> WorkflowOrchestratorBackend:
    backend = ctx.workflow_orchestrator
    if backend is None:
        raise RuntimeError("workflow_orchestrator_not_configured")
    return backend


def workflow_trigger(ctx: ToolWiringContext, params: WorkflowTriggerInput) -> WorkflowTriggerOutput:
    handle = _require_orchestrator(ctx).trigger_run(
        params.workflow_id,
        parameters=dict(params.parameters),
    )
    return WorkflowTriggerOutput(run_id=handle.run_id, status=handle.status, url=handle.url)


def workflow_poll(ctx: ToolWiringContext, params: WorkflowPollInput) -> WorkflowPollOutput:
    status = _require_orchestrator(ctx).poll_status(params.run_id)
    return WorkflowPollOutput(
        run_id=status.run_id,
        status=status.status,
        conclusion=status.conclusion,
        logs_uri=status.logs_uri,
    )


def workflow_fetch_logs(ctx: ToolWiringContext, params: WorkflowFetchLogsInput) -> WorkflowFetchLogsOutput:
    logs = _require_orchestrator(ctx).fetch_logs(params.run_id, tail_lines=params.tail_lines)
    return WorkflowFetchLogsOutput(run_id=params.run_id, logs=logs)
