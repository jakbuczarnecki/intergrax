# © Artur Czarnecki. All rights reserved.

"""PostgreSQL task checkpoint store — contract negatives (P4-A)."""

from __future__ import annotations

import pytest

from tests.qualification.dist_scale_x._p2_support import pg_port_open

pytestmark = [
    pytest.mark.skipif(not pg_port_open(), reason="PostgreSQL qual infra unavailable"),
]


@pytest.mark.unit
def test_pg_checkpoint_create_and_cas_stale() -> None:
    from intergrax.integrations.providers.relational_store.postgresql.config import (
        PostgreSQLIntegrationConfig,
    )
    from intergrax.runtime.long_running.checkpoint_revision import StaleCheckpointWriteError
    from intergrax.runtime.long_running.postgresql_task_checkpoint_store import (
        PostgreSQLTaskCheckpointStore,
    )
    from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
    from intergrax.runtime.task.task import Task, TaskState
    from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions
    from intergrax.contracts.execution_identity import (
        bind_active_execution_identity,
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        reset_active_execution_identity,
    )

    config = PostgreSQLIntegrationConfig.from_env()
    config = config.model_copy(update={"tenant_schema": "dist_scale_x_p4_unit"})
    store = PostgreSQLTaskCheckpointStore(config=config)
    task = Task(
        tenant_id="t-pg-1",
        user_id="u1",
        message="pg qual",
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
        cp = LongRunningCoordinator.persist_checkpoint(task, store, run_id=run_id, attempt_id=attempt_id)
        assert cp.revision == 1
        task.runtime.orchestration.checkpoint_revision = 0
        with pytest.raises(StaleCheckpointWriteError):
            LongRunningCoordinator.persist_checkpoint(
                task,
                store,
                run_id=run_id,
                attempt_id=attempt_id,
                progress_message="stale-cas",
            )
    finally:
        reset_active_execution_identity(token)
