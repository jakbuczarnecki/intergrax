# © Artur Czarnecki. All rights reserved.

"""LKW background ingest worker handler contract (LKW.4D)."""

from __future__ import annotations

import json
from typing import Protocol

from intergrax.contracts.execution_identity import AttemptId, RunId, TaskId
from intergrax.queueing.contracts.task_queue import TaskRequest
from intergrax.queueing.contracts.task_queue import TaskResult as QueueTaskResult
from intergrax.queueing.contracts.task_queue import TaskStatus
from intergrax.runtime.task.task import Task, TaskContext, TaskResult as RuntimeTaskResult

from local_workspace_application.background_ingest.contracts import (
    LKW_BACKGROUND_INGEST_TASK_NAME,
    LkwBackgroundIngestJob,
    decode_background_ingest_job,
)
from local_workspace_application.workspaces.document_indexing import extract_ingest_summary

LKW_BACKGROUND_INGEST_CAPABILITY = "local.workspace.index"
LKW_BACKGROUND_INGEST_AGENT_ID = "local_indexer"


class BackgroundIngestTaskRunner(Protocol):
    async def run_task(
        self,
        task: Task,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
    ) -> RuntimeTaskResult:
        ...


def decode_background_ingest_task_request(request: TaskRequest) -> LkwBackgroundIngestJob:
    if request.task_name != LKW_BACKGROUND_INGEST_TASK_NAME:
        raise ValueError("unexpected_background_ingest_task_name")
    job = decode_background_ingest_job(request.payload)
    if request.tenant_id and job.tenant_id != request.tenant_id:
        raise ValueError("background_ingest_tenant_mismatch")
    return job


def build_background_ingest_runtime_task(
    request: TaskRequest,
    job: LkwBackgroundIngestJob,
    *,
    task_id: TaskId,
) -> Task:
    metadata: dict[str, object] = {
        "_hydrate_legacy": False,
        "source_paths": list(job.source_paths),
        "collection_id": job.collection_id,
        "workspace_id": job.workspace_id,
        "tenant_id": job.tenant_id,
        "chunking_strategy_id": "recursive",
        "requested_by": job.requested_by,
        "background_ingest_schema_version": job.schema_version,
        "background_ingest_task_name": LKW_BACKGROUND_INGEST_TASK_NAME,
        "background_ingest_idempotency_key": request.idempotency_key,
        "background_ingest_priority": job.priority,
        "background_ingest_broker_run_id": request.run_id,
    }
    if job.correlation_id is not None:
        metadata["background_ingest_correlation_id"] = job.correlation_id
    if job.reason is not None:
        metadata["background_ingest_reason"] = job.reason
    if job.change_token is not None:
        metadata["background_ingest_change_token"] = job.change_token

    return Task(
        task_id=task_id,
        tenant_id=job.tenant_id,
        user_id=job.requested_by or "background_ingest",
        agent_id=LKW_BACKGROUND_INGEST_AGENT_ID,
        message=f"Background ingest for workspace {job.workspace_id}",
        context=TaskContext(capability=LKW_BACKGROUND_INGEST_CAPABILITY),
        metadata=metadata,
    )


def _runtime_result_output(
    result: RuntimeTaskResult,
    *,
    request: TaskRequest,
    job: LkwBackgroundIngestJob,
) -> bytes:
    payload = {
        "task_id": result.task_id,
        "run_id": result.run_id,
        "state": result.state.value,
        "agent_id": result.agent_id,
        "answer": result.answer,
        "metadata": result.metadata,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _failed_queue_result(error: BaseException) -> QueueTaskResult:
    message = str(error).strip() or error.__class__.__name__
    return QueueTaskResult(
        status=TaskStatus.FAILED,
        error_message=message,
        attempts=1,
    )


async def handle_background_ingest_task_request(
    request: TaskRequest,
    runner: BackgroundIngestTaskRunner,
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
) -> QueueTaskResult:
    try:
        job = decode_background_ingest_task_request(request)
        task = build_background_ingest_runtime_task(request, job, task_id=task_id)
        runtime_result = await runner.run_task(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
        )
    except Exception as exc:
        return _failed_queue_result(exc)

    ingest_summary = extract_ingest_summary(runtime_result)
    if not ingest_summary.get("used"):
        reason = str(ingest_summary.get("reason") or "background_ingest_not_indexed")
        return QueueTaskResult(
            status=TaskStatus.FAILED,
            error_message=reason,
            attempts=1,
        )

    return QueueTaskResult(
        status=TaskStatus.SUCCEEDED,
        output=_runtime_result_output(runtime_result, request=request, job=job),
        attempts=1,
    )
