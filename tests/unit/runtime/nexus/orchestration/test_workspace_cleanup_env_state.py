# © Artur Czarnecki. All rights reserved.

"""APP-CON-8 — clear isolation refs from task env state after cleanup."""

from __future__ import annotations

import pytest

from intergrax.applications.contracts.environment_state import (
    APP_ENV_STATE_RUNTIME_KEY,
    ApplicationEnvironmentState,
    SandboxIsolationRef,
    WorkspaceIsolationRef,
    seed_application_environment_state,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.runtime.nexus.orchestration.workspace_cleanup import clear_isolation_refs_in_task_env_state
from intergrax.runtime.task.task import Task

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_clear_isolation_refs_in_task_env_state() -> None:
    task = Task(
        tenant_id="tenant-1",
        user_id="user-1",
        message="cleanup",
        metadata=seed_application_environment_state(
            app_id="legal",
            profile_id="legal.product",
            execution_mode=ExecutionMode.STRICT,
            task_id="task-clear-1",
        ),
    )
    raw = task.metadata[APP_ENV_STATE_RUNTIME_KEY]
    assert isinstance(raw, dict)
    state = ApplicationEnvironmentState.model_validate(raw)
    state = state.model_copy(
        update={
            "shadow_workspace": WorkspaceIsolationRef(
                workspace_id="ws-1",
                tenant_id="tenant-1",
                task_id="task-clear-1",
            ),
            "sandbox_session": SandboxIsolationRef(
                session_id="sb-1",
                tenant_id="tenant-1",
                task_id="task-clear-1",
            ),
        }
    )
    task.metadata[APP_ENV_STATE_RUNTIME_KEY] = state.model_dump(mode="json")

    clear_isolation_refs_in_task_env_state(task)

    cleared = ApplicationEnvironmentState.model_validate(task.metadata[APP_ENV_STATE_RUNTIME_KEY])
    assert cleared.shadow_workspace is None
    assert cleared.sandbox_session is None
