# © Artur Czarnecki. All rights reserved.

"""APP-CON-6 — RunArtifactBundle rollup on task completion."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.run_artifact_bundle_builder import (
    build_run_artifact_bundle,
    stage_application_artifact,
)
from intergrax.applications.contracts.application_artifacts import (
    APPLICATION_ARTIFACTS_STAGING_KEY,
    ApplicationArtifactRef,
    RUN_ARTIFACT_BUNDLE_METADATA_KEY,
)
from intergrax.applications.contracts.environment_state import seed_application_environment_state
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.sandbox.sandbox_runtime import SANDBOX_SESSION_ID_KEY
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskIsolationOptions
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.runtime.workspace.shadow_workspace import SHADOW_WORKSPACE_ID_KEY

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_stage_application_artifact_accumulates_in_task_metadata() -> None:
    task = Task(tenant_id="tenant-1", user_id="user-1", message="export")
    stage_application_artifact(
        task,
        ApplicationArtifactRef(
            artifact_id="art-export-1",
            kind="export",
            uri="file:///tmp/report.json",
            task_id=task.task_id,
            owner_app_id="research",
            tenant_id="tenant-1",
        ),
    )
    assert APPLICATION_ARTIFACTS_STAGING_KEY in task.metadata
    assert len(task.metadata[APPLICATION_ARTIFACTS_STAGING_KEY]) == 1


def test_build_run_artifact_bundle_collects_staged_and_workspace(tmp_path) -> None:
    shadow_manager = ShadowWorkspaceManager(root=tmp_path / "shadow")
    sandbox_manager = SandboxSessionManager(root=tmp_path / "sandbox")
    task = Task(
        tenant_id="tenant-1",
        user_id="user-1",
        message="bundle",
        options=TaskExecutionOptions(
            isolation=TaskIsolationOptions(shadow_workspace=True, sandbox=True),
        ),
        metadata=seed_application_environment_state(
            app_id="research",
            profile_id="research.product",
            execution_mode=ExecutionMode.STRICT,
            task_id="task-bundle-1",
        ),
    )
    stage_application_artifact(
        task,
        ApplicationArtifactRef(
            artifact_id="art-export-1",
            kind="export",
            uri="file:///tmp/report.json",
            task_id=task.task_id,
            owner_app_id="research",
            tenant_id="tenant-1",
        ),
    )
    workspace = shadow_manager.open_or_create(tenant_id=task.tenant_id, task_id=task.task_id)
    workspace.write_text("outputs/summary.md", "# Summary")
    session = sandbox_manager.open_or_create(tenant_id=task.tenant_id, task_id=task.task_id)
    session.execute("write_file", {"path": "result.txt", "content": "ok"})
    executions = [
        AgentExecutionResult(
            agent_id="research",
            run_id="run-1",
            status=AgentExecutionStatus.COMPLETED,
            structured_data={
                SHADOW_WORKSPACE_ID_KEY: workspace.workspace_id,
                SANDBOX_SESSION_ID_KEY: session.session_id,
            },
        )
    ]
    bundle = build_run_artifact_bundle(
        task=task,
        graph_id="graph-1",
        executions=executions,
        shadow_manager=shadow_manager,
        sandbox_manager=sandbox_manager,
    )
    assert bundle.schema_version == RUN_ARTIFACT_BUNDLE_METADATA_KEY
    assert len(bundle.application) == 1
    assert bundle.application[0].owner_app_id == "research"
    assert len(bundle.workspace) == 1
    assert bundle.workspace[0].relative_path == "outputs/summary.md"
    assert len(bundle.sandbox) == 1
    assert bundle.sandbox[0].relative_path == "result.txt"
