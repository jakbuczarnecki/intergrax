# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker queue serialization for Nexus Task v2 (§41, J.3)."""

from __future__ import annotations

import json
from typing import Any, Dict

from intergrax.fastapi_core.execution.models import ExecutionRequest

NEXUS_TASK_V2_LOGICAL_NAME = "nexus.task.v2"
WORKER_PAYLOAD_SCHEMA = "nexus_worker_payload.v1"


def encode_execution_request(request: ExecutionRequest) -> bytes:
    """Serialize ExecutionRequest for Celery worker transport."""
    payload: Dict[str, Any] = {
        "schema_version": WORKER_PAYLOAD_SCHEMA,
        "run_id": request.run_id,
        "tenant_id": request.tenant_id,
        "user_id": request.user_id,
        "input_payload": dict(request.input_payload),
        "metadata": dict(request.metadata),
        "config": dict(request.config),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_execution_request(payload: bytes) -> ExecutionRequest:
    """Deserialize ExecutionRequest from worker transport bytes."""
    raw = json.loads(payload.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("worker payload must be a JSON object")
    return ExecutionRequest(
        run_id=str(raw["run_id"]),
        tenant_id=str(raw["tenant_id"]),
        user_id=raw.get("user_id"),
        input_payload=dict(raw.get("input_payload") or {}),
        metadata=dict(raw.get("metadata") or {}),
        config=dict(raw.get("config") or {}),
    )
