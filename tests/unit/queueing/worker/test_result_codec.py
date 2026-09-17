# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.queueing.worker.result_codec import (
    decode_host_task_result_payload,
    decode_logical_task_result,
    encode_logical_task_result,
    nexus_result_payload_from_envelope,
)
from intergrax.runtime.task.nexus_worker_execution import NexusTaskWorkerOutput
from intergrax.tools.execution_models import ToolExecutionResult
from pydantic import BaseModel

pytestmark = pytest.mark.unit


def test_encode_decode_logical_task_result_roundtrip() -> None:
    result = ToolExecutionResult.ok(
        NexusTaskWorkerOutput(result_payload={"answer": "ok", "state": "completed"})
    )
    envelope = decode_logical_task_result(encode_logical_task_result(result))
    assert envelope["success"] is True
    payload = nexus_result_payload_from_envelope(envelope)
    assert payload == {"answer": "ok", "state": "completed"}


def test_nexus_result_payload_from_failed_envelope() -> None:
    assert nexus_result_payload_from_envelope({"success": False, "output": None}) is None


def test_decode_host_task_result_payload_bytes_envelope() -> None:
    result = ToolExecutionResult.ok(
        NexusTaskWorkerOutput(result_payload={"answer": "ok", "state": "completed"})
    )
    raw = encode_logical_task_result(result)
    payload = decode_host_task_result_payload(raw)
    assert payload == {"answer": "ok", "state": "completed"}


def test_decode_host_task_result_payload_tool_execution_result() -> None:
    result = ToolExecutionResult.ok(
        NexusTaskWorkerOutput(result_payload={"answer": "direct"})
    )
    payload = decode_host_task_result_payload(result)
    assert payload == {"answer": "direct"}


def test_decode_host_task_result_payload_rejects_duck_typed_result_payload() -> None:
    class Fake:
        result_payload = {"answer": "forged"}

    assert decode_host_task_result_payload(Fake()) is None


def test_decode_host_task_result_payload_rejects_non_nexus_tool_output() -> None:
    class OtherOutput(BaseModel):
        value: str = "x"

    wrapped = ToolExecutionResult.ok(OtherOutput())
    assert decode_host_task_result_payload(wrapped) is None


def test_decode_host_task_result_payload_rejects_bare_nexus_json_without_envelope() -> None:
    raw = NexusTaskWorkerOutput(result_payload={"answer": "bare"}).model_dump_json().encode("utf-8")
    assert decode_host_task_result_payload(raw) is None
