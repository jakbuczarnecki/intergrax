# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from unittest.mock import Mock

import pytest

from intergrax.queueing.contracts.task_queue import (
    TaskHandle,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from intergrax.queueing.providers.celery.celery_task_queue import CeleryTaskQueue

pytestmark = pytest.mark.unit

class DummyAsyncResult:
    def __init__(self, state, result=None, retries=0):
        self.state = state
        self.result = result
        self.retries = retries


def test_enqueue_sends_generic_task():
    mock_app = Mock()
    mock_task = Mock()
    mock_result = Mock()
    mock_result.id = "task-123"
    mock_task.apply_async.return_value = mock_result
    mock_app.tasks = {"intergrax.execute": mock_task}

    queue = CeleryTaskQueue(app=mock_app)

    request = TaskRequest(
        task_name="logical.task",
        tenant_id="t1",
        run_id="r1",
        payload=b"data",
        idempotency_key="abc",
    )

    handle = queue.enqueue(request)

    mock_task.apply_async.assert_called_once()

    _, kwargs = mock_task.apply_async.call_args

    assert kwargs["kwargs"]["logical_task_name"] == "logical.task"
    assert handle.task_id == "task-123"
    assert handle.provider == "celery"


@pytest.mark.parametrize(
    "celery_state,expected_status",
    [
        ("PENDING", TaskStatus.PENDING),
        ("STARTED", TaskStatus.RUNNING),
        ("RETRY", TaskStatus.RUNNING),
        ("SUCCESS", TaskStatus.SUCCEEDED),
        ("FAILURE", TaskStatus.FAILED),
        ("REVOKED", TaskStatus.FAILED),
    ],
)
def test_get_status_mapping(celery_state, expected_status):
    mock_app = Mock()
    mock_app.AsyncResult.return_value = DummyAsyncResult(state=celery_state)

    queue = CeleryTaskQueue(app=mock_app)

    handle = TaskHandle(task_id="id", provider="celery")

    status = queue.get_status(handle)

    assert status == expected_status


def test_get_result_success():
    mock_app = Mock()
    mock_app.AsyncResult.return_value = DummyAsyncResult(
        state="SUCCESS",
        result=b"ok",
        retries=2,
    )

    queue = CeleryTaskQueue(app=mock_app)

    handle = TaskHandle(task_id="id", provider="celery")

    result = queue.get_result(handle)

    assert isinstance(result, TaskResult)
    assert result.status == TaskStatus.SUCCEEDED
    assert result.output == b"ok"
    assert result.error_message is None
    assert result.attempts == 2


def test_get_result_failure():
    mock_app = Mock()
    mock_app.AsyncResult.return_value = DummyAsyncResult(
        state="FAILURE",
        result=Exception("boom"),
        retries=1,
    )

    queue = CeleryTaskQueue(app=mock_app)

    handle = TaskHandle(task_id="id", provider="celery")

    result = queue.get_result(handle)

    assert isinstance(result, TaskResult)
    assert result.status == TaskStatus.FAILED
    assert result.output is None
    assert "boom" in result.error_message
    assert result.attempts == 1


def test_get_result_not_finished():
    mock_app = Mock()
    mock_app.AsyncResult.return_value = DummyAsyncResult(
        state="STARTED"
    )

    queue = CeleryTaskQueue(app=mock_app)

    handle = TaskHandle(task_id="id", provider="celery")

    result = queue.get_result(handle)

    assert result is None