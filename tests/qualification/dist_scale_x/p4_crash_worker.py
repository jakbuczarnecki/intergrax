# © Artur Czarnecki. All rights reserved.

"""Crash-point worker for P4 dimension G (hard kill at sync boundaries)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tests.qualification.dist_scale_x._p4_support import ensure_qual_pg_schema
from tests.qualification.dist_scale_x.p4_persistence_op import execute_p4_persistence_op, mint_qual_task_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--sync-file", required=True)
    parser.add_argument("--state-file", required=True)
    args = parser.parse_args()
    sync = Path(args.sync_file)
    state = Path(args.state_file)
    ensure_qual_pg_schema(args.dsn, args.schema)
    pad = "c" * 1024
    boundary = args.boundary
    if boundary == "G-01":
        sync.write_text("armed", encoding="utf-8")
        time.sleep(300)
        return
    task_id = mint_qual_task_id()
    if boundary == "G-02":
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

        config = PostgreSQLIntegrationConfig.from_env().model_copy(
            update={"dsn": args.dsn, "tenant_schema": args.schema},
        )
        store = PostgreSQLTaskCheckpointStore(config=config)
        task = Task(
            tenant_id=args.tenant_id,
            user_id="g02",
            message="g02",
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
        state.write_text(
            json.dumps({"boundary": boundary, "task_id": saved.task_id}),
            encoding="utf-8",
        )
        sync.write_text("after_f01", encoding="utf-8")
        time.sleep(300)
        return
    if boundary in {"G-03", "G-04", "G-05", "G-06"}:
        result = execute_p4_persistence_op(
            dsn=args.dsn,
            schema=args.schema,
            tenant_id=args.tenant_id,
            op_index=1,
            payload_pad=pad,
            task_id=task_id,
        )
        state.write_text(
            json.dumps({"boundary": boundary, "task_id": result.task_id, "kind": result.kind}),
            encoding="utf-8",
        )
        sync.write_text(f"after_{boundary}", encoding="utf-8")
        time.sleep(300)


if __name__ == "__main__":
    main()
