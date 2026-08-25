# © Artur Czarnecki. All rights reserved.

"""TaskExecutionRegistry handler for LKW background ingest (LKW.4D / LKW.4E)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from typing import Optional, Protocol

from pydantic import BaseModel, Field

from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.runtime.task.task import TaskResult as RuntimeTaskResult
from intergrax.tools.execution_models import ToolExecutionResult
from local_workspace_application.background_ingest.contracts import (
    LKW_BACKGROUND_INGEST_TASK_NAME,
    LkwBackgroundIngestJob,
    background_ingest_idempotency_key,
    decode_background_ingest_job,
)
from local_workspace_application.background_ingest.handler import (
    handle_background_ingest_task_request,
)


class BackgroundIngestTaskRunner(Protocol):
    async def run_task(self, task: object) -> RuntimeTaskResult:
        ...


class BackgroundIngestWorkerOutput(BaseModel):
    """Worker handler output returned through Tier-0 queue plane."""

    answer: str | None = None
    agent_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    schema_version: str = "lkw.background_ingest_worker.v1"


def _run_coro_sync(coro: object) -> object:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def make_background_ingest_worker_handler(
    runner: BackgroundIngestTaskRunner,
):
    """Build a TaskExecutionRegistry handler for ``lkw.background_ingest.v1``."""

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: Optional[str] = None,
        execution_identity: BackgroundExecutionIdentity,
    ) -> ToolExecutionResult[BackgroundIngestWorkerOutput]:
        _ = execution_identity
        try:
            job = decode_background_ingest_job(payload)
            if job.tenant_id != tenant_id:
                return ToolExecutionResult.fail(
                    "background_ingest_tenant_mismatch",
                    "background_ingest_tenant_mismatch",
                )
            resolved_idempotency = idempotency_key or background_ingest_idempotency_key(job)
            request = TaskRequest(
                tenant_id=tenant_id,
                run_id=run_id,
                task_name=LKW_BACKGROUND_INGEST_TASK_NAME,
                payload=payload,
                idempotency_key=resolved_idempotency,
            )
            queue_result = _run_coro_sync(
                handle_background_ingest_task_request(request, runner),
            )
            if queue_result.status != TaskStatus.SUCCEEDED:
                message = queue_result.error_message or queue_result.status.value
                return ToolExecutionResult.fail("background_ingest_failed", message)

            decoded: dict[str, object] = {}
            if queue_result.output:
                parsed = json.loads(queue_result.output.decode("utf-8"))
                if isinstance(parsed, dict):
                    decoded = parsed

            return ToolExecutionResult.ok(
                BackgroundIngestWorkerOutput(
                    answer=str(decoded.get("answer") or ""),
                    agent_id=str(decoded.get("agent_id") or "") or None,
                    metadata=dict(decoded.get("metadata") or {}),
                )
            )
        except Exception as exc:  # noqa: BLE001 - worker plane normalizes failures
            return ToolExecutionResult.fail(type(exc).__name__, str(exc))

    return handler


def register_background_ingest_worker_handler(
    registry: TaskExecutionRegistry,
    runner: BackgroundIngestTaskRunner,
    *,
    logical_task_name: str = LKW_BACKGROUND_INGEST_TASK_NAME,
) -> None:
    registry.register(logical_task_name, make_background_ingest_worker_handler(runner))
