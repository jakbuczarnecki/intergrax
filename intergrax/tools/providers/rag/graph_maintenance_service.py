# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.workflow_orchestrator import WorkflowOrchestratorBackend
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.tools.providers.rag.graph_maintenance_contracts import (
    RagScheduleGraphMaintenanceJobInput,
    RagScheduleGraphMaintenanceJobOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

RAG_SCHEDULE_GRAPH_MAINTENANCE_JOB_TOOL_ID = "rag.schedule_graph_maintenance_job"

_ACTIVE_RUN_STATUSES = frozenset({"pending", "queued", "running", "scheduled"})


def _require_orchestrator(ctx: ToolWiringContext) -> WorkflowOrchestratorBackend:
    backend = ctx.workflow_orchestrator
    if backend is None:
        raise RuntimeError("workflow_orchestrator_not_configured")
    return backend


def _build_idempotency_key(params: RagScheduleGraphMaintenanceJobInput) -> str:
    if params.idempotency_key and params.idempotency_key.strip():
        return params.idempotency_key.strip()
    tenant = (params.tenant_id or "default").strip()
    workspace = (params.workspace_id or "default").strip()
    return f"graph-maint:{params.mode}:{tenant}:{workspace}"


def _find_active_run(
    backend: WorkflowOrchestratorBackend,
    *,
    workflow_id: str,
    idempotency_key: str,
) -> str | None:
    for item in backend.list_runs(workflow_id=workflow_id, limit=50):
        if item.metadata.get("idempotency_key") != idempotency_key:
            continue
        if item.status.lower() in _ACTIVE_RUN_STATUSES:
            return item.run_id
    return None


def perform_rag_schedule_graph_maintenance_job(
    ctx: ToolWiringContext,
    params: RagScheduleGraphMaintenanceJobInput,
) -> RagScheduleGraphMaintenanceJobOutput:
    try:
        backend = _require_orchestrator(ctx)
    except RuntimeError:
        return RagScheduleGraphMaintenanceJobOutput(used=False, reason="workflow_orchestrator_not_configured")

    profile = ctx.rag_profile or RagProfile()
    if not profile.graph_rag_enabled:
        return RagScheduleGraphMaintenanceJobOutput(used=False, reason="graph_rag_disabled")

    workflow_id = (params.workflow_id or profile.graph_maintenance_workflow_id).strip()
    if not workflow_id:
        return RagScheduleGraphMaintenanceJobOutput(used=False, reason="workflow_id_not_configured")

    idempotency_key = _build_idempotency_key(params)
    existing_run_id = _find_active_run(
        backend,
        workflow_id=workflow_id,
        idempotency_key=idempotency_key,
    )
    if existing_run_id is not None:
        status = backend.poll_status(existing_run_id)
        return RagScheduleGraphMaintenanceJobOutput(
            used=True,
            run_id=existing_run_id,
            status=status.status or "pending",
            url=status.logs_uri,
            idempotency_key=idempotency_key,
            reason="idempotent_reuse",
        )

    workflow_params = {
        "job_type": "rag.graph_maintenance",
        "mode": params.mode,
        "idempotency_key": idempotency_key,
    }
    if params.tenant_id:
        workflow_params["tenant_id"] = params.tenant_id
    if params.workspace_id:
        workflow_params["workspace_id"] = params.workspace_id

    handle = backend.trigger_run(workflow_id, parameters=workflow_params)
    return RagScheduleGraphMaintenanceJobOutput(
        used=True,
        run_id=handle.run_id,
        status=handle.status,
        url=handle.url,
        idempotency_key=idempotency_key,
        reason="scheduled",
    )
