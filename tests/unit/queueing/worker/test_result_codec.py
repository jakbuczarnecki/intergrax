# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.queueing.worker.result_codec import (
    decode_logical_task_result,
    encode_logical_task_result,
    nexus_result_payload_from_envelope,
)
from intergrax.runtime.task.nexus_worker_execution import NexusTaskWorkerOutput
from intergrax.tools.execution_models import ToolExecutionResult

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
