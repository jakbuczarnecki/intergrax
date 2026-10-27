# © Artur Czarnecki. All rights reserved.

"""H: SIGKILL after full persist, recover in new process with continuation."""

from __future__ import annotations

import argparse
import json
import os
import time
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

from tests.qualification.dist_scale_x.p4_pg_state_inspection import inspect_p4_durable_state
from tests.qualification.dist_scale_x.p4_staged_persistence import (
    BOUNDARY_SYNC,
    execute_staged_boundary,
    handle_to_json,
    handle_from_json,
)
from tests.qualification.dist_scale_x.p4_persistence_op import mint_qual_task_id


def _write_phase(dsn: str, schema: str, tenant_id: str, sync: Path, state: Path) -> None:
    task_id = mint_qual_task_id()
    handle = execute_staged_boundary(
        dsn=dsn,
        schema=schema,
        tenant_id=tenant_id,
        boundary="G-06",
        task_id=task_id,
        subject="h-recovery",
    )
    state.write_text(handle_to_json(handle), encoding="utf-8")
    sync.write_text(BOUNDARY_SYNC["G-06"], encoding="utf-8")


def _recover_phase(state: Path, out: Path) -> None:
    handle = handle_from_json(state.read_text(encoding="utf-8"))
    before = inspect_p4_durable_state(
        dsn=handle.dsn,
        schema=handle.schema,
        tenant_id=handle.tenant_id,
        task_id=handle.task_id,
        run_id=handle.run_id,
        decision_id=handle.decision_id,
        decision_subject=handle.decision_subject,
        event_id=handle.event_id,
    )
    config = PostgreSQLIntegrationConfig.from_env().model_copy(
        update={"dsn": handle.dsn, "tenant_schema": handle.schema},
    )
    store = PostgreSQLTaskCheckpointStore(config=config)
    task = Task(
        tenant_id=handle.tenant_id,
        user_id="h-recover",
        message="recovery-continue",
        state=TaskState.WAITING_FOR_HUMAN,
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    task.task_id = handle.task_id
    task.runtime.orchestration.checkpoint_revision = before.f01_revision or 1
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
            progress_message="recovery-continue",
        )
    finally:
        reset_active_execution_identity(token)
    after = inspect_p4_durable_state(
        dsn=handle.dsn,
        schema=handle.schema,
        tenant_id=handle.tenant_id,
        task_id=handle.task_id,
        run_id=handle.run_id,
        decision_id=handle.decision_id,
        decision_subject=handle.decision_subject,
        event_id=handle.event_id,
    )
    out.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "before_revision": before.f01_revision,
                "after_revision": saved.revision,
                "f03_present": after.f03_present,
                "f09_present": after.f09_present,
                "f07_event_present": after.f07_event_present,
            },
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--tenant-id", default="tenant-h")
    parser.add_argument("--mode", choices=("write", "recover"), required=True)
    parser.add_argument("--sync-file", default="")
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    state = Path(args.state_file)
    if args.mode == "write":
        sync = Path(args.sync_file)
        _write_phase(args.dsn, args.schema, args.tenant_id, sync, state)
        time.sleep(300)
        return
    _recover_phase(state, Path(args.out))


if __name__ == "__main__":
    main()
