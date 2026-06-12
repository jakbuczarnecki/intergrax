# © Artur Czarnecki. All rights reserved.

"""Build Plane A ``RunArtifactBundle`` rollup on task completion (APP-CON-6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from intergrax.applications.contracts.application_artifacts import (
    APPLICATION_ARTIFACTS_STAGING_KEY,
    ApplicationArtifactRef,
    RunArtifactBundle,
    SandboxArtifactRef,
    WorkspaceArtifactRef,
)
from intergrax.applications.contracts.environment_state import (
    APP_ENV_STATE_RUNTIME_KEY,
    ApplicationEnvironmentState,
)
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.sandbox.sandbox_runtime import SANDBOX_SESSION_ID_KEY
from intergrax.runtime.task.task import Task
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.runtime.workspace.shadow_workspace import SHADOW_WORKSPACE_ID_KEY


def stage_application_artifact(task: Task, ref: ApplicationArtifactRef) -> None:
    """Append a host-produced artifact ref to task metadata for final rollup."""
    raw = task.metadata.get(APPLICATION_ARTIFACTS_STAGING_KEY, [])
    staged: list[dict[str, Any]] = []
    if isinstance(raw, list):
        staged = [item for item in raw if isinstance(item, dict)]
    staged.append(ref.model_dump(mode="json"))
    task.metadata[APPLICATION_ARTIFACTS_STAGING_KEY] = staged
    task.sync_metadata()


def _owner_app_id(task: Task) -> str:
    env_blob = task.metadata.get(APP_ENV_STATE_RUNTIME_KEY)
    if isinstance(env_blob, dict):
        state = ApplicationEnvironmentState.from_runtime_state({APP_ENV_STATE_RUNTIME_KEY: env_blob})
        if state is not None and state.app_id:
            return state.app_id
    app_id = task.metadata.get("app_id")
    if isinstance(app_id, str) and app_id.strip():
        return app_id.strip()
    return "unknown"


def _staged_application_artifacts(task: Task, *, owner_app_id: str) -> list[ApplicationArtifactRef]:
    raw = task.metadata.get(APPLICATION_ARTIFACTS_STAGING_KEY, [])
    if not isinstance(raw, list):
        return []
    refs: list[ApplicationArtifactRef] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        payload.setdefault("task_id", task.task_id)
        payload.setdefault("tenant_id", task.tenant_id)
        payload.setdefault("owner_app_id", owner_app_id)
        refs.append(ApplicationArtifactRef.model_validate(payload))
    return refs


def _workspace_artifact_refs(
    task: Task,
    executions: list[AgentExecutionResult],
    *,
    shadow_manager: ShadowWorkspaceManager,
) -> list[WorkspaceArtifactRef]:
    if not task.options.isolation.shadow_workspace:
        return []

    workspace = None
    if executions:
        workspace_id = executions[-1].structured_data.get(SHADOW_WORKSPACE_ID_KEY)
        if workspace_id:
            workspace = shadow_manager.get(str(workspace_id))
    if workspace is None:
        workspace = shadow_manager.open_or_create(tenant_id=task.tenant_id, task_id=task.task_id)
    if workspace is None or not workspace.exists_on_disk():
        return []

    refs: list[WorkspaceArtifactRef] = []
    for artifact in workspace.list_artifacts():
        refs.append(
            WorkspaceArtifactRef(
                artifact_id=artifact.artifact_id,
                workspace_id=workspace.workspace_id,
                relative_path=artifact.relative_path,
                uri=Path(workspace.root, artifact.relative_path).resolve().as_uri(),
                size_bytes=artifact.size_bytes,
                sha256=artifact.sha256,
                task_id=task.task_id,
                tenant_id=task.tenant_id,
            )
        )
    return refs


def _sandbox_artifact_refs(
    task: Task,
    executions: list[AgentExecutionResult],
    *,
    sandbox_manager: SandboxSessionManager,
) -> list[SandboxArtifactRef]:
    if not task.options.isolation.sandbox:
        return []

    session = None
    if executions:
        session_id = executions[-1].structured_data.get(SANDBOX_SESSION_ID_KEY)
        if session_id:
            session = sandbox_manager.get(str(session_id))
    if session is None:
        session = sandbox_manager.open_or_create(tenant_id=task.tenant_id, task_id=task.task_id)
    if session is None or not session.exists_on_disk():
        return []

    refs: list[SandboxArtifactRef] = []
    for path in sorted(session.root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(session.root).as_posix()
        data = path.read_bytes()
        refs.append(
            SandboxArtifactRef(
                artifact_id=f"sbox_{session.session_id}_{rel.replace('/', '_')}",
                session_id=session.session_id,
                relative_path=rel,
                uri=path.resolve().as_uri(),
                size_bytes=len(data),
                sha256="",
                task_id=task.task_id,
                tenant_id=task.tenant_id,
            )
        )
    return refs


def build_run_artifact_bundle(
    *,
    task: Task,
    graph_id: str,
    executions: list[AgentExecutionResult],
    shadow_manager: ShadowWorkspaceManager,
    sandbox_manager: SandboxSessionManager,
) -> RunArtifactBundle:
    """Materialize artifact rollup from staged metadata and active isolation sessions."""
    owner_app_id = _owner_app_id(task)
    return RunArtifactBundle(
        task_id=task.task_id,
        graph_id=graph_id,
        application=_staged_application_artifacts(task, owner_app_id=owner_app_id),
        workspace=_workspace_artifact_refs(
            task,
            executions,
            shadow_manager=shadow_manager,
        ),
        sandbox=_sandbox_artifact_refs(
            task,
            executions,
            sandbox_manager=sandbox_manager,
        ),
    )
