# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from intergrax.contracts.acp_metadata_keys import AcpStructuredDataKey
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.agent_run_enums import AgentRunStatus
from intergrax.contracts.run_artifact_bundle import RUN_ARTIFACT_BUNDLE_METADATA_KEY
from intergrax.contracts.task_artifacts import RunArtifactBundle, WorkspaceArtifactRef
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.runtime.task.task_metadata_keys import TaskResultMetadataKey
from local_workspace_application.host.lifecycle import LocalWorkspaceHostLifecycle
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor
from local_workspace_application.serving.fastapi_router import LocalWorkspaceRunService
from local_workspace_application.serving.run_artifact_metadata import (
    ensure_run_artifact_bundle_metadata,
    extract_run_artifact_bundle,
    find_synthesize_workspace_artifact,
    run_artifact_bundle_payload_is_safe,
)
from local_workspace_application.serving.schemas import LocalWorkspaceRunRequestV1


def _synthesize_bundle_payload() -> dict[str, object]:
    return RunArtifactBundle(
        task_id="task-synth",
        graph_id="graph-synth",
        workspace=[
            WorkspaceArtifactRef(
                artifact_id="art-1",
                workspace_id="shadow-ws-1",
                relative_path="synthesis-draft.md",
                uri="file:///tmp/shadow/synthesis-draft.md",
                task_id="task-synth",
                tenant_id="tenant-1",
            )
        ],
    ).model_dump(mode="json")


@pytest.mark.unit
def test_extract_run_artifact_bundle_from_task_result_metadata() -> None:
    bundle_payload = _synthesize_bundle_payload()
    task_result = TaskResult(
        task_id="task-synth",
        run_id="run-synth",
        state=TaskState.COMPLETED,
        answer="done",
        agent_id="local_synthesizer",
        metadata={TaskResultMetadataKey.RUN_ARTIFACT_BUNDLE: bundle_payload},
    )

    bundle = extract_run_artifact_bundle(task_result)

    assert bundle is not None
    assert bundle.schema_version == RUN_ARTIFACT_BUNDLE_METADATA_KEY
    assert len(bundle.workspace) == 1
    assert bundle.workspace[0].relative_path == "synthesis-draft.md"


@pytest.mark.unit
def test_extract_run_artifact_bundle_from_application_run_summary_nested_metadata() -> None:
    bundle_payload = _synthesize_bundle_payload()
    task_result = TaskResult(
        task_id="task-synth",
        run_id="run-synth",
        state=TaskState.COMPLETED,
        answer="done",
        agent_id="local_synthesizer",
        metadata={
            TaskResultMetadataKey.APPLICATION_RUN_SUMMARY: {
                "schema_version": "application_run_summary.v1",
                "metadata": {RUN_ARTIFACT_BUNDLE_METADATA_KEY: bundle_payload},
            }
        },
    )

    bundle = extract_run_artifact_bundle(task_result)

    assert bundle is not None
    assert bundle.workspace[0].artifact_id == "art-1"


@pytest.mark.unit
def test_find_synthesize_workspace_artifact_matches_path_and_ref() -> None:
    bundle = RunArtifactBundle.model_validate(_synthesize_bundle_payload())

    by_path = find_synthesize_workspace_artifact(
        bundle,
        artifact_path="synthesis-draft.md",
        artifact_ref=None,
    )
    by_ref = find_synthesize_workspace_artifact(
        bundle,
        artifact_path=None,
        artifact_ref="shadow-ws-1/art-1",
    )

    assert by_path is not None
    assert by_ref is not None
    assert by_path.artifact_id == "art-1"
    assert by_ref.relative_path == "synthesis-draft.md"


@pytest.mark.unit
def test_run_artifact_bundle_payload_is_safe_rejects_raw_content_fields() -> None:
    payload = _synthesize_bundle_payload()
    assert run_artifact_bundle_payload_is_safe(payload) is True

    unsafe = dict(payload)
    workspace = list(unsafe["workspace"])  # type: ignore[index]
    workspace[0] = {**workspace[0], "content": "secret draft body"}
    unsafe["workspace"] = workspace
    assert run_artifact_bundle_payload_is_safe(unsafe) is False


@pytest.mark.unit
def test_ensure_run_artifact_bundle_metadata_promotes_nested_bundle() -> None:
    bundle_payload = _synthesize_bundle_payload()
    task_result = TaskResult(
        task_id="task-synth",
        run_id="run-synth",
        state=TaskState.COMPLETED,
        answer="done",
        agent_id="local_synthesizer",
        metadata={
            TaskResultMetadataKey.APPLICATION_RUN_SUMMARY: {
                "schema_version": "application_run_summary.v1",
                "metadata": {RUN_ARTIFACT_BUNDLE_METADATA_KEY: bundle_payload},
            }
        },
    )
    metadata: dict[str, object] = {}

    ensure_run_artifact_bundle_metadata(metadata, task_result=task_result)

    assert TaskResultMetadataKey.RUN_ARTIFACT_BUNDLE in metadata
    assert metadata[TaskResultMetadataKey.RUN_ARTIFACT_BUNDLE]["workspace"][0]["relative_path"] == "synthesis-draft.md"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_task_exposes_platform_bundle_and_lkw_synthesize_diagnostic() -> None:
    bundle_payload = _synthesize_bundle_payload()
    execution = AgentExecutionResult(
        agent_id="local_synthesizer",
        run_id="run-api",
        status=AgentExecutionStatus.COMPLETED,
        summary="answer",
        structured_data={
            AcpStructuredDataKey.TRACE_SUMMARY: {
                "total_steps": 1,
                "step_diagnostics": {
                    "lkw.synthesize_summary.v1": {
                        "write_status": "write_complete",
                        "shadow_write": True,
                        "source_evidence_count": 1,
                        "artifact_path": "synthesis-draft.md",
                        "artifact_ref": "shadow-ws-1/art-1",
                    }
                },
            }
        },
    )
    task_result = TaskResult(
        task_id="task-api",
        run_id="run-api",
        state=TaskState.COMPLETED,
        answer="answer",
        agent_id="local_synthesizer",
        execution_result=execution,
        metadata={
            TaskResultMetadataKey.RUN_ARTIFACT_BUNDLE: bundle_payload,
            TaskResultMetadataKey.APPLICATION_RUN_SUMMARY: {
                "schema_version": "application_run_summary.v1",
                "terminal_status": AgentRunStatus.SUCCEEDED.value,
            },
        },
    )
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.set_executor_available(True)
    lifecycle.transition_to_ready()
    executor = AsyncMock(spec=LocalWorkspaceTaskExecutor)
    executor.execute = AsyncMock(return_value=task_result)
    service = LocalWorkspaceRunService(task_executor=executor, default_agent_id="local_synthesizer")

    response = await service.run_task(
        LocalWorkspaceRunRequestV1(
            message="synthesize draft",
            capability="local.workspace.synthesize",
            metadata={"shadow_workspace": True},
        )
    )

    bundle = response.metadata[TaskResultMetadataKey.RUN_ARTIFACT_BUNDLE]
    evidence = response.metadata["lkw_evidence.v1"]
    synth_diag = evidence["diagnostics"]["lkw.synthesize_summary.v1"]

    assert bundle["schema_version"] == RUN_ARTIFACT_BUNDLE_METADATA_KEY
    assert len(bundle["workspace"]) == 1
    assert run_artifact_bundle_payload_is_safe(bundle) is True
    assert "content" not in bundle["workspace"][0]

    matched = find_synthesize_workspace_artifact(
        RunArtifactBundle.model_validate(bundle),
        artifact_path=synth_diag.get("artifact_path"),
        artifact_ref=synth_diag.get("artifact_ref"),
    )
    assert matched is not None
    assert matched.relative_path == "synthesis-draft.md"
    assert synth_diag["shadow_write"] is True
