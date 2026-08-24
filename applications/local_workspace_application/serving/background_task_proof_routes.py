# © Artur Czarnecki. All rights reserved.

"""Local/dev-only LKW Kafka background-task platform proof routes (LKW.4E)."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, status
from pydantic import BaseModel

from intergrax.fastapi_core.config import ApiEnvironment
from intergrax.queueing.contracts.task_queue import TaskStatus
from intergrax.tools.providers.message_bus.contracts import (
    MessageBusGetResultInput,
    MessageBusGetStatusInput,
)
from intergrax.tools.providers.message_bus.service import message_bus_get_result, message_bus_get_status
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.background_ingest.contracts import (
    LKW_BACKGROUND_INGEST_TASK_NAME,
    LkwBackgroundIngestJob,
)
from local_workspace_application.background_ingest.enqueue import enqueue_background_ingest_job
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

_PROOF_TENANT_ID = "lkw-background-proof"
_PROOF_WORKSPACE_ID = "lkw-background-proof"
_PROOF_COLLECTION_ID = "local_workspace"
_DEFAULT_MARKER_PREFIX = "LKW_BACKGROUND_TASK_PROOF"


class LocalWorkspaceBackgroundTaskProofEnqueueRequestV1(BaseModel):
    run_id: str = ""
    correlation_id: str = ""
    marker: str = ""


class LocalWorkspaceBackgroundTaskProofEnqueueResponseV1(BaseModel):
    proof_result: Literal["PASS", "FAIL"]
    proof_kind: Literal["platform_background_task"] = "platform_background_task"
    phase: Literal["enqueue"] = "enqueue"
    task_name: str = LKW_BACKGROUND_INGEST_TASK_NAME
    message_bus_provider: str
    enqueue_mode: Literal["real_provider"] = "real_provider"
    worker_execution: Literal["asynchronous"] = "asynchronous"
    mock_queue: Literal[False] = False
    run_id: str
    correlation_id: str
    marker: str
    source_path: str
    collection_id: str = _PROOF_COLLECTION_ID
    task_id: str = ""
    provider: str = ""
    tenant_id: str = _PROOF_TENANT_ID
    initial_task_status: str = ""
    enqueue_latency_ms: int = 0


class LocalWorkspaceBackgroundTaskProofStatusResponseV1(BaseModel):
    proof_kind: Literal["platform_background_task"] = "platform_background_task"
    phase: Literal["status"] = "status"
    task_name: str = LKW_BACKGROUND_INGEST_TASK_NAME
    task_id: str
    provider: str
    tenant_id: str
    task_status: str
    completed: bool = False
    has_result: bool = False
    error_message: str = ""
    runtime_task_id: str = ""
    runtime_run_id: str = ""
    broker_run_id: str = ""
    change_token: str = ""
    idempotency_key: str = ""


def _proof_endpoint_enabled(settings: LocalWorkspaceBackendSettings) -> bool:
    return settings.environment in {ApiEnvironment.DEV, ApiEnvironment.STAGE}


def _require_message_bus(ctx: ToolWiringContext) -> tuple[ToolWiringContext, str]:
    bus = ctx.message_bus
    if bus is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="background_task_proof_disabled: message_bus_not_configured",
        )
    provider = getattr(bus, "_provider_name", None) or getattr(bus, "provider", None)
    if provider is None:
        provider = "kafka"
    return ctx, str(provider)


def _proof_docs_root() -> Path:
    for candidate in (
        Path("/data/user_docs"),
        Path("applications/local_workspace_application/sample_docs"),
    ):
        if candidate.is_dir():
            return candidate
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="background_task_proof_docs_unavailable",
    )


def _write_proof_document(marker: str) -> tuple[str, Path]:
    docs_root = _proof_docs_root()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    file_name = f"lkw_background_task_proof_{timestamp}.txt"
    path = docs_root / file_name
    path.write_text(
        "\n".join(
            [
                marker,
                "Intergrax LKW background task platform proof document.",
                "Indexed exclusively through message_bus.enqueue -> Kafka -> worker.",
            ]
        ),
        encoding="utf-8",
    )
    container_path = f"/data/user_docs/{file_name}"
    return container_path, path


async def enqueue_local_workspace_background_task_proof(
    *,
    settings: LocalWorkspaceBackendSettings,
    wiring_context: ToolWiringContext,
    run_id: str,
    correlation_id: str,
    marker: str,
) -> LocalWorkspaceBackgroundTaskProofEnqueueResponseV1:
    ctx, provider = _require_message_bus(wiring_context)
    source_path, _ = _write_proof_document(marker)
    job = LkwBackgroundIngestJob(
        tenant_id=_PROOF_TENANT_ID,
        workspace_id=_PROOF_WORKSPACE_ID,
        collection_id=_PROOF_COLLECTION_ID,
        source_paths=(source_path,),
        requested_by="lkw.background_task_proof",
        run_id=run_id,
        correlation_id=correlation_id,
        reason="lkw.4e.platform_proof",
    )
    started = time.perf_counter()
    try:
        output = enqueue_background_ingest_job(ctx, job, run_id=run_id)
    except Exception as exc:  # noqa: BLE001 - proof endpoint surfaces operator-safe failure
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"background_task_proof_enqueue_failed: {exc.__class__.__name__}",
        ) from exc
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    status_output = message_bus_get_status(
        ctx,
        MessageBusGetStatusInput(
            task_id=output.task_id,
            provider=output.provider,
            tenant_id=output.tenant_id,
        ),
    )
    return LocalWorkspaceBackgroundTaskProofEnqueueResponseV1(
        proof_result="PASS",
        message_bus_provider=provider,
        run_id=run_id,
        correlation_id=correlation_id,
        marker=marker,
        source_path=source_path,
        collection_id=_PROOF_COLLECTION_ID,
        task_id=output.task_id,
        provider=output.provider,
        initial_task_status=status_output.status.value,
        enqueue_latency_ms=elapsed_ms,
    )


def _decode_background_task_result_identity(
  result_output_base64: str,
) -> dict[str, str]:
    if not result_output_base64.strip():
        return {}
    try:
        import base64
        import json

        raw = base64.b64decode(result_output_base64.encode("ascii"))
        parsed = json.loads(raw.decode("utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    execution_identity = parsed.get("execution_identity")
    identity_payload: dict[str, object] = {}
    if isinstance(execution_identity, dict):
        identity_payload = execution_identity
    return {
        "runtime_task_id": str(
            identity_payload.get("runtime_task_id") or parsed.get("runtime_task_id") or ""
        ),
        "runtime_run_id": str(
            identity_payload.get("runtime_run_id") or parsed.get("runtime_run_id") or ""
        ),
        "broker_run_id": str(identity_payload.get("broker_run_id") or ""),
        "change_token": str(identity_payload.get("change_token") or ""),
        "idempotency_key": str(identity_payload.get("idempotency_key") or ""),
    }


def inspect_local_workspace_background_task_status(
    *,
    wiring_context: ToolWiringContext,
    task_id: str,
    provider: str,
    tenant_id: str,
) -> LocalWorkspaceBackgroundTaskProofStatusResponseV1:
    ctx, _ = _require_message_bus(wiring_context)
    status_output = message_bus_get_status(
        ctx,
        MessageBusGetStatusInput(task_id=task_id, provider=provider, tenant_id=tenant_id),
    )
    result_output = message_bus_get_result(
        ctx,
        MessageBusGetResultInput(task_id=task_id, provider=provider, tenant_id=tenant_id),
    )
    terminal = status_output.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED}
    identity_fields = _decode_background_task_result_identity(result_output.output_base64)
    return LocalWorkspaceBackgroundTaskProofStatusResponseV1(
        task_id=task_id,
        provider=provider,
        tenant_id=tenant_id,
        task_status=status_output.status.value,
        completed=terminal,
        has_result=result_output.completed,
        error_message=result_output.error_message or "",
        runtime_task_id=identity_fields.get("runtime_task_id", ""),
        runtime_run_id=identity_fields.get("runtime_run_id", ""),
        broker_run_id=identity_fields.get("broker_run_id", ""),
        change_token=identity_fields.get("change_token", ""),
        idempotency_key=identity_fields.get("idempotency_key", ""),
    )


def mount_local_workspace_background_task_proof_routes(
    app: FastAPI,
    *,
    settings: LocalWorkspaceBackendSettings,
    wiring_context: ToolWiringContext,
    prefix: str,
) -> None:
    if not _proof_endpoint_enabled(settings):
        return

    router = APIRouter(prefix=prefix, tags=["local_workspace_proof"])

    @router.post(
        "/proof/background-task/enqueue",
        response_model=LocalWorkspaceBackgroundTaskProofEnqueueResponseV1,
    )
    async def background_task_enqueue_proof(
        body: LocalWorkspaceBackgroundTaskProofEnqueueRequestV1 | None = None,
    ) -> LocalWorkspaceBackgroundTaskProofEnqueueResponseV1:
        request = body or LocalWorkspaceBackgroundTaskProofEnqueueRequestV1()
        run_id = request.run_id.strip() or f"lkw-bg-proof-{uuid.uuid4().hex[:12]}"
        correlation_id = request.correlation_id.strip() or f"corr-{uuid.uuid4().hex[:12]}"
        marker = request.marker.strip() or (
            f"{_DEFAULT_MARKER_PREFIX}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        )
        return await enqueue_local_workspace_background_task_proof(
            settings=settings,
            wiring_context=wiring_context,
            run_id=run_id,
            correlation_id=correlation_id,
            marker=marker,
        )

    @router.get(
        "/proof/background-task/status/{provider}/{task_id}",
        response_model=LocalWorkspaceBackgroundTaskProofStatusResponseV1,
    )
    async def background_task_status_proof(
        provider: str,
        task_id: str,
        tenant_id: str = _PROOF_TENANT_ID,
    ) -> LocalWorkspaceBackgroundTaskProofStatusResponseV1:
        return inspect_local_workspace_background_task_status(
            wiring_context=wiring_context,
            task_id=task_id,
            provider=provider,
            tenant_id=tenant_id,
        )

    app.include_router(router)
