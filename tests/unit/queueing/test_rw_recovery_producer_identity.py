# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-E NT-RW-REC-08 (EX-D7-03, F-D7-09): the adopted enqueue sites carry ``TaskRequest.identity``.

Structural pin over the three adopted ``TaskRequest(...)`` constructor sites plus a behavioural check
of the retention-sweep builder. The three sites outside this ticket's write set are named in one
explicit residual tuple, so EX-D7-03 stays visibly incomplete until they are ruled (Q-RC-04).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from intergrax.queueing.contracts.task_queue import TaskExecutionIdentity

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[3]

ADOPTED_ENQUEUE_SITES: tuple[str, ...] = (
    "applications/local_workspace_application/background_ingest/worker_handler.py",
    "intergrax/memory/retention_sweep.py",
    "intergrax/queueing/providers/document_store/document_store_task_queue.py",
)

# Residual (EX-D7-03 not complete): constructor sites outside every train ticket's write set.
RESIDUAL_ENQUEUE_SITES_WITHOUT_IDENTITY: tuple[str, ...] = (
    "intergrax/runtime/task/queued_host_task_execution_adapter.py",
    "intergrax/runtime/vendor_knowledge/sync_jobs.py",
    "intergrax/tools/providers/message_bus/service.py",
)


def _task_request_calls(relative_path: str) -> list[ast.Call]:
    tree = ast.parse((_REPO_ROOT / relative_path).read_text(encoding="utf-8"), filename=relative_path)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "TaskRequest"
    ]


def test_rw_rec_08_adopted_enqueue_sites_pass_identity() -> None:
    missing: list[str] = []
    for relative_path in ADOPTED_ENQUEUE_SITES:
        calls = _task_request_calls(relative_path)
        assert calls, f"no TaskRequest(...) constructor found in {relative_path}"
        for call in calls:
            if not any(keyword.arg == "identity" for keyword in call.keywords):
                missing.append(f"{relative_path}:{call.lineno}")
    assert missing == [], f"TaskRequest built without identity=: {missing}"

    for relative_path in RESIDUAL_ENQUEUE_SITES_WITHOUT_IDENTITY:
        assert _task_request_calls(relative_path), f"residual site {relative_path} no longer builds TaskRequest"
    assert not set(ADOPTED_ENQUEUE_SITES) & set(RESIDUAL_ENQUEUE_SITES_WITHOUT_IDENTITY)

    from intergrax.memory.retention_sweep import memory_retention_sweep_task_request

    identity = TaskExecutionIdentity(
        task_id="task_" + "a" * 32,
        run_id="run_" + "b" * 32,
        attempt_id="attempt_" + "c" * 32,
        execution_id="exec_" + "d" * 32,
    )
    carried = memory_retention_sweep_task_request(
        tenant_id="t-rec08",
        run_id="transport-run-1",
        user_id="user-1",
        identity=identity,
    )
    assert carried.identity == identity
    assert memory_retention_sweep_task_request(tenant_id="t-rec08", run_id="r", user_id="u").identity is None
