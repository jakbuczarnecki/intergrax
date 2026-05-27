# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import uuid4

from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState

TASK_PAYLOAD_VERSION = 1


def new_run_id() -> str:
    return f"run_{uuid4().hex}"


def task_from_runtime_request(
    runtime_req: RuntimeRequest,
    *,
    tenant_id: str,
    user_id: str,
    run_id: Optional[str] = None,
    capability: Optional[str] = None,
) -> Task:
    """Build a Nexus Task from a RuntimeRequest (HTTP / eval intake)."""
    resolved_run_id = run_id or runtime_req.metadata.get("run_id") or new_run_id()
    cap = capability or runtime_req.metadata.get("capability")
    return Task(
        task_id=resolved_run_id,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=runtime_req.session_id,
        agent_id=runtime_req.agent_id,
        message=runtime_req.message or "",
        context=TaskContext(capability=cap),
        metadata={
            "source": "runtime_request",
            "runtime_metadata": dict(runtime_req.metadata),
        },
    )


def task_from_execution_request(request: ExecutionRequest) -> Task:
    """Deserialize Task from FastAPI Core ExecutionRequest payload."""
    payload = request.input_payload or {}
    task_payload = payload.get("task")
    if isinstance(task_payload, dict):
        return Task.model_validate(task_payload)

    message = str(payload.get("message", ""))
    capability = payload.get("capability")
    agent_id = payload.get("agent_id")
    session_id = payload.get("session_id")

    return Task(
        task_id=request.run_id,
        tenant_id=request.tenant_id,
        user_id=request.user_id or "",
        session_id=str(session_id) if session_id else None,
        agent_id=str(agent_id) if agent_id else None,
        message=message,
        context=TaskContext(
            capability=str(capability) if capability else None,
        ),
        metadata=dict(payload.get("metadata") or {}),
    )


def task_to_execution_payload(task: Task) -> Dict[str, Any]:
    """Serialize Task for RunService / worker dispatch."""
    return {
        "version": TASK_PAYLOAD_VERSION,
        "task": task.model_dump(mode="json"),
    }


def task_result_to_payload(result: TaskResult) -> Dict[str, Any]:
    """Serialize TaskResult for RunStore.result_payload."""
    payload: Dict[str, Any] = {
        "task_id": result.task_id,
        "run_id": result.run_id,
        "state": result.state.value,
        "answer": result.answer,
        "agent_id": result.agent_id,
        "metadata": dict(result.metadata),
    }
    if result.execution_result is not None:
        payload["execution_result"] = result.execution_result.model_dump(mode="json")
    return payload


def runtime_request_with_run_id(request: RuntimeRequest, run_id: str) -> RuntimeRequest:
    """Attach unified run_id to RuntimeRequest metadata for RuntimeEngine."""
    meta = dict(request.metadata)
    meta["run_id"] = run_id
    return RuntimeRequest(
        agent_id=request.agent_id,
        user_id=request.user_id,
        session_id=request.session_id,
        message=request.message,
        attachments=list(request.attachments),
        workspace_id=request.workspace_id,
        tenant_id=request.tenant_id,
        metadata=meta,
        instructions=request.instructions,
        history_compression_strategy=request.history_compression_strategy,
        max_output_tokens=request.max_output_tokens,
    )
