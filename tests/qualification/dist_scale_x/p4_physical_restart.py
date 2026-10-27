# © Artur Czarnecki. All rights reserved.

"""P4 dimension G/H physical slices (WL-5 scope)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.qualification.dist_scale_x._p4_support import qual_pg_schema, require_p4_pg_infra, resolve_pg_dsn

pytestmark = [pytest.mark.qualification, pytest.mark.integration]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYTHON = sys.executable


def test_p4_h_durable_restart_reload_checkpoint() -> None:
    """H-01: new OS process reloads F01 checkpoint from same PG DSN/schema."""
    require_p4_pg_infra()
    dsn = resolve_pg_dsn()
    schema = qual_pg_schema("p4-h-restart")
    module = "tests.qualification.dist_scale_x.p4_restart_worker"
    out = _REPO_ROOT / ".tmp" / "session" / "p4-h-restart" / "state.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            _PYTHON,
            "-m",
            module,
            "--dsn",
            dsn,
            "--schema",
            schema,
            "--out",
            str(out),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.exists()
    payload = out.read_text(encoding="utf-8")
    assert "revision" in payload


def test_p4_g_cross_process_cas_conflict() -> None:
    """G-02 class: stale CAS visible across independent connections."""
    require_p4_pg_infra()
    from intergrax.integrations.providers.relational_store.postgresql.config import (
        PostgreSQLIntegrationConfig,
    )
    from intergrax.runtime.long_running.checkpoint_revision import StaleCheckpointWriteError
    from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
    from intergrax.runtime.long_running.postgresql_task_checkpoint_store import (
        PostgreSQLTaskCheckpointStore,
    )
    from intergrax.runtime.task.task import Task, TaskState
    from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions
    from intergrax.contracts.execution_identity import (
        bind_active_execution_identity,
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        reset_active_execution_identity,
    )

    schema = qual_pg_schema("p4-g-cas")
    config = PostgreSQLIntegrationConfig.from_env().model_copy(update={"tenant_schema": schema})
    store_a = PostgreSQLTaskCheckpointStore(config=config)
    store_b = PostgreSQLTaskCheckpointStore(config=config)
    task = Task(
        tenant_id="tenant-g",
        user_id="u1",
        message="g-cas",
        state=TaskState.WAITING_FOR_HUMAN,
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    try:
        LongRunningCoordinator.persist_checkpoint(task, store_a, run_id=run_id, attempt_id=attempt_id)
        task.runtime.orchestration.checkpoint_revision = 0
        with pytest.raises(StaleCheckpointWriteError):
            LongRunningCoordinator.persist_checkpoint(
                task,
                store_b,
                run_id=run_id,
                attempt_id=attempt_id,
                progress_message="cross-process-stale",
            )
    finally:
        reset_active_execution_identity(token)
