# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import task_to_execution_payload
from intergrax.runtime.task.worker_payload import (
    decode_execution_request,
    encode_execution_request,
)

pytestmark = pytest.mark.unit


def test_encode_decode_execution_request_preserves_task_payload() -> None:
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hello worker",
        context=TaskContext(capability="echo.basic"),
    )
    request = ExecutionRequest(
        run_id="run_worker_unit",
        tenant_id="t1",
        user_id="u1",
        input_payload=task_to_execution_payload(task),
        metadata={"request_id": "req-1"},
    )
    restored = decode_execution_request(encode_execution_request(request))
    assert restored.run_id == "run_worker_unit"
    assert restored.metadata["request_id"] == "req-1"
    assert restored.input_payload["task"]["context"]["capability"] == "echo.basic"
