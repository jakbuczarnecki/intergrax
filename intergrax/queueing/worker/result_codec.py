# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""JSON codec for logical-task worker results (Tier-0 transport bytes)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel

from intergrax.tools.execution_models import ToolExecutionError, ToolExecutionResult

BytesLike = Union[bytes, str]


def encode_logical_task_result(result: ToolExecutionResult[BaseModel]) -> bytes:
    """Serialize ToolExecutionResult to UTF-8 JSON bytes for Celery/backend transport."""
    payload: Dict[str, Any] = {
        "success": result.success,
        "output": result.output.model_dump() if result.output is not None else None,
        "error": (
            {
                "error_code": result.error.error_code,
                "error_message": result.error.error_message,
            }
            if result.error is not None
            else None
        ),
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def decode_logical_task_result(data: BytesLike) -> Dict[str, Any]:
    """Deserialize worker result envelope from transport bytes."""
    text = data.decode("utf-8") if isinstance(data, bytes) else data
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("worker result envelope must be a JSON object")
    return parsed


def nexus_result_payload_from_envelope(envelope: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract Nexus ``result_payload`` from a decoded worker envelope."""
    if not envelope.get("success"):
        return None
    output = envelope.get("output")
    if not isinstance(output, dict):
        return None
    result_payload = output.get("result_payload")
    if not isinstance(result_payload, dict):
        return None
    return dict(result_payload)


def worker_result_bytes_from_transport(raw: object) -> Optional[bytes]:
    """Normalize Celery AsyncResult payloads to UTF-8 JSON bytes."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, str):
        return raw.encode("utf-8")
    return None
