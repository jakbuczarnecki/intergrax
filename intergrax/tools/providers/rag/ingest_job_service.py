# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pathlib import Path
from typing import Any

from intergrax.integrations.contracts.workflow_orchestrator import WorkflowOrchestratorBackend
from intergrax.rag.ingest.ingest_policy import build_ingest_idempotency_key
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.tools.providers.rag.ingest_contracts import (
    RagScheduleIngestJobInput,
    RagScheduleIngestJobOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

RAG_SCHEDULE_INGEST_JOB_TOOL_ID = "rag.schedule_ingest_job"

_ACTIVE_RUN_STATUSES = frozenset({"pending", "queued", "running", "scheduled"})


def _require_orchestrator(ctx: ToolWiringContext) -> WorkflowOrchestratorBackend:
    backend = ctx.workflow_orchestrator
    if backend is None:
        raise RuntimeError("workflow_orchestrator_not_configured")
    return backend


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


def _workflow_parameters(
    *,
    source_path: str,
    idempotency_key: str,
    params: RagScheduleIngestJobInput,
    metadata: dict[str, Any],
) -> dict[str, str]:
    workflow_params: dict[str, str] = {
        "source_path": source_path,
        "idempotency_key": idempotency_key,
        "job_type": "rag.ingest",
    }
    if params.session_id is not None:
        workflow_params["session_id"] = params.session_id
    if params.user_id is not None:
        workflow_params["user_id"] = params.user_id
    if params.tenant_id is not None:
        workflow_params["tenant_id"] = params.tenant_id
    if params.workspace_id is not None:
        workflow_params["workspace_id"] = params.workspace_id
    for key, value in metadata.items():
        workflow_params[f"meta.{key}"] = str(value)
    return workflow_params


def perform_rag_schedule_ingest_job(
    ctx: ToolWiringContext,
    params: RagScheduleIngestJobInput,
) -> RagScheduleIngestJobOutput:
    path = Path(params.source_path)
    if not path.exists():
        return RagScheduleIngestJobOutput(used=False, reason="source_not_found")

    try:
        backend = _require_orchestrator(ctx)
    except RuntimeError:
        return RagScheduleIngestJobOutput(used=False, reason="workflow_orchestrator_not_configured")

    profile = ctx.rag_profile or RagProfile()
    workflow_id = (params.workflow_id or profile.async_ingest_workflow_id).strip()
    if not workflow_id:
        return RagScheduleIngestJobOutput(used=False, reason="workflow_id_not_configured")

    idempotency_key = build_ingest_idempotency_key(
        source_path=str(path),
        tenant_id=params.tenant_id,
        workspace_id=params.workspace_id,
        explicit_key=params.idempotency_key,
    )

    existing_run_id = _find_active_run(
        backend,
        workflow_id=workflow_id,
        idempotency_key=idempotency_key,
    )
    if existing_run_id is not None:
        status = backend.poll_status(existing_run_id)
        return RagScheduleIngestJobOutput(
            used=True,
            run_id=existing_run_id,
            status=status.status or "pending",
            url=status.logs_uri,
            idempotency_key=idempotency_key,
            reason="idempotent_reuse",
        )

    workflow_params = _workflow_parameters(
        source_path=str(path.resolve()),
        idempotency_key=idempotency_key,
        params=params,
        metadata=dict(params.metadata),
    )
    handle = backend.trigger_run(workflow_id, parameters=workflow_params)
    return RagScheduleIngestJobOutput(
        used=True,
        run_id=handle.run_id,
        status=handle.status,
        url=handle.url,
        idempotency_key=idempotency_key,
        reason="scheduled",
    )
