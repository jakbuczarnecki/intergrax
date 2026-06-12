# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import List, Optional

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.sandbox.sandbox_runtime import SANDBOX_SESSION_ID_KEY
from intergrax.runtime.task.task import Task
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.applications.contracts.environment_state import (
    APP_ENV_STATE_RUNTIME_KEY,
    ApplicationEnvironmentState,
)
from intergrax.runtime.task.task import Task
from intergrax.runtime.workspace.shadow_workspace import SHADOW_WORKSPACE_ID_KEY


def clear_isolation_refs_in_task_env_state(task: Task) -> None:
    """Drop shadow/sandbox handles from persisted ``app_env_state.v1`` after task cleanup."""
    raw = task.metadata.get(APP_ENV_STATE_RUNTIME_KEY)
    if raw is None:
        return
    if isinstance(raw, ApplicationEnvironmentState):
        state = raw
    elif isinstance(raw, dict):
        state = ApplicationEnvironmentState.model_validate(raw)
    else:
        return
    if state.shadow_workspace is None and state.sandbox_session is None:
        return
    cleared = state.model_copy(update={"shadow_workspace": None, "sandbox_session": None})
    task.metadata[APP_ENV_STATE_RUNTIME_KEY] = cleared.model_dump(mode="json")
    task.sync_metadata()


def cleanup_shadow_for_task(
    task: Task,
    executions: List[AgentExecutionResult],
    *,
    shadow_manager: ShadowWorkspaceManager,
) -> None:
    iso = task.options.isolation
    if not iso.shadow_workspace or not iso.shadow_workspace_cleanup:
        return
    workspace_id: Optional[str] = None
    if executions:
        workspace_id = executions[-1].structured_data.get(SHADOW_WORKSPACE_ID_KEY)
    if workspace_id:
        shadow_manager.cleanup(str(workspace_id))
    else:
        shadow_manager.cleanup_for_task(
            tenant_id=task.tenant_id,
            task_id=task.task_id,
        )


def cleanup_sandbox_for_task(
    task: Task,
    executions: List[AgentExecutionResult],
    *,
    sandbox_manager: SandboxSessionManager,
) -> None:
    iso = task.options.isolation
    if not iso.sandbox or not iso.sandbox_cleanup:
        return
    session_id: Optional[str] = None
    if executions:
        session_id = executions[-1].structured_data.get(SANDBOX_SESSION_ID_KEY)
    if session_id:
        sandbox_manager.cleanup(str(session_id))
    else:
        sandbox_manager.cleanup_for_task(
            tenant_id=task.tenant_id,
            task_id=task.task_id,
        )
