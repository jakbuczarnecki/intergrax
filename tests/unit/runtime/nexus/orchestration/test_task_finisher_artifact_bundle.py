# © Artur Czarnecki. All rights reserved.

"""APP-CON-6 — task finisher attaches run_artifact_bundle.v1."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.run_artifact_bundle_builder import stage_application_artifact
from intergrax.applications.contracts.application_artifacts import (
    ApplicationArtifactRef,
    RUN_ARTIFACT_BUNDLE_METADATA_KEY,
)
from intergrax.applications.contracts.environment_state import seed_application_environment_state
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.orchestration.task_finisher import build_nexus_task_result
from intergrax.runtime.nexus.response.final_response_composer import FinalResponseComposer
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_metadata_keys import TaskResultMetadataKey
from intergrax.runtime.task.task_trace import TaskTraceEmitter
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_build_nexus_task_result_attaches_run_artifact_bundle(tmp_path) -> None:
    shadow_manager = ShadowWorkspaceManager(root=tmp_path / "shadow")
    sandbox_manager = SandboxSessionManager(root=tmp_path / "sandbox")
    task = Task(
        tenant_id="tenant-1",
        user_id="user-1",
        message="finish",
        state=TaskState.COMPLETED,
        metadata=seed_application_environment_state(
            app_id="legal",
            profile_id="legal.product",
            execution_mode=ExecutionMode.STRICT,
            task_id="task-finisher-1",
        ),
    )
    stage_application_artifact(
        task,
        ApplicationArtifactRef(
            artifact_id="art-legal-1",
            kind="memo",
            uri="file:///tmp/memo.pdf",
            task_id=task.task_id,
            owner_app_id="legal",
            tenant_id="tenant-1",
        ),
    )
    executions = [
        AgentExecutionResult(
            agent_id="legal",
            run_id="run-legal-1",
            status=AgentExecutionStatus.COMPLETED,
            summary="done",
        )
    ]
    result = build_nexus_task_result(
        task,
        TaskTraceEmitter(run_id=task.task_id),
        answer="done",
        executions=executions,
        validation=ValidationResult(valid=True),
        plan=None,
        retry_records=[],
        graph_id="graph-legal",
        composer=FinalResponseComposer(),
        event_bus=RuntimeEventBus(),
        shadow_manager=shadow_manager,
        sandbox_manager=sandbox_manager,
    )
    assert TaskResultMetadataKey.RUN_ARTIFACT_BUNDLE in result.metadata
    bundle = result.metadata[TaskResultMetadataKey.RUN_ARTIFACT_BUNDLE]
    assert bundle["schema_version"] == RUN_ARTIFACT_BUNDLE_METADATA_KEY
    assert len(bundle["application"]) == 1

    summary = result.metadata[TaskResultMetadataKey.APPLICATION_RUN_SUMMARY]
    assert RUN_ARTIFACT_BUNDLE_METADATA_KEY in summary["metadata"]
    assert summary["metadata"][RUN_ARTIFACT_BUNDLE_METADATA_KEY]["application"][0]["artifact_id"] == "art-legal-1"
