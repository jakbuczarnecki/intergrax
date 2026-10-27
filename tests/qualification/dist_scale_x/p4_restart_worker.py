# © Artur Czarnecki. All rights reserved.

"""Write checkpoint in isolated process; verify reload in parent (P4 H)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.postgresql_task_checkpoint_store import (
    PostgreSQLTaskCheckpointStore,
)
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = PostgreSQLIntegrationConfig.from_env().model_copy(
        update={"dsn": args.dsn, "tenant_schema": args.schema},
    )
    store = PostgreSQLTaskCheckpointStore(config=config)
    task = Task(
        tenant_id="tenant-h",
        user_id="u1",
        message="restart",
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
        saved = LongRunningCoordinator.persist_checkpoint(
            task,
            store,
            run_id=run_id,
            attempt_id=attempt_id,
        )
    finally:
        reset_active_execution_identity(token)
    reloaded = PostgreSQLTaskCheckpointStore(config=config).get_latest(saved.task_id, saved.tenant_id)
    assert reloaded is not None
    Path(args.out).write_text(
        json.dumps({"revision": reloaded.revision, "task_id": reloaded.task_id}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
