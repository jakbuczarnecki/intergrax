# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.acp_metadata_keys import AcpStructuredDataKey
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.agent_run_enums import AgentRunStatus, StepNextAction
from intergrax.contracts.agent_run_trace import AgentRunTrace, AgentStepRecord, AgentStepStatus
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.runtime.task.task_metadata_keys import TaskResultMetadataKey
from local_workspace_application.serving.evidence_slice import (
    LKW_EVIDENCE_SCHEMA_VERSION,
    build_lkw_evidence_slice,
    build_lkw_evidence_slice_from_step_diagnostics,
    collect_lkw_diagnostics_from_trace,
)
from local_workspace_application.serving.run_metadata import attach_lkw_evidence_metadata


def _step_with_diagnostics(diagnostics: dict[str, object]) -> AgentStepRecord:
    return AgentStepRecord(
        step_id="step-0",
        step_index=0,
        status=AgentStepStatus.SUCCEEDED,
        next_action=StepNextAction.CONTINUE,
        state_version=1,
        diagnostics=diagnostics,
    )


@pytest.mark.unit
def test_collect_lkw_diagnostics_extracts_typed_schemas() -> None:
    trace = AgentRunTrace(
        run_id="run-1",
        steps=[
            _step_with_diagnostics(
                {
                    "lkw.index_summary.v1": {"accepted_count": 2, "chunk_count": 5},
                    "platform.noise.v1": {"ignored": True},
                }
            ),
            _step_with_diagnostics(
                {
                    "lkw.search_summary.v1": {
                        "num_results": 3,
                        "evidence_count": 3,
                        "source_refs": ["docs/a.md"],
                    }
                }
            ),
            _step_with_diagnostics(
                {
                    "lkw.synthesize_summary.v1": {
                        "write_status": "write_complete",
                        "shadow_write": True,
                        "source_evidence_count": 3,
                        "artifact_path": "synthesis-draft.md",
                        "artifact_ref": "shadow-ws-1/art-1",
                    }
                }
            ),
        ],
    )
    diagnostics = collect_lkw_diagnostics_from_trace(trace)
    assert set(diagnostics) == {
        "lkw.index_summary.v1",
        "lkw.search_summary.v1",
        "lkw.synthesize_summary.v1",
    }
    assert diagnostics["lkw.search_summary.v1"]["num_results"] == 3
    synth = diagnostics["lkw.synthesize_summary.v1"]
    assert synth["artifact_path"] == "synthesis-draft.md"
    assert synth["artifact_ref"] == "shadow-ws-1/art-1"


@pytest.mark.unit
def test_collect_lkw_diagnostics_ignores_unrelated_and_handles_missing() -> None:
    trace = AgentRunTrace(run_id="run-empty", steps=[])
    assert collect_lkw_diagnostics_from_trace(trace) == {}

    trace_with_noise = AgentRunTrace(
        run_id="run-noise",
        steps=[_step_with_diagnostics({"acp.tool.v1": {"tool_id": "rag.search"}})],
    )
    assert collect_lkw_diagnostics_from_trace(trace_with_noise) == {}


@pytest.mark.unit
def test_build_lkw_evidence_slice_redacts_unsafe_search_fields() -> None:
    trace = AgentRunTrace(
        run_id="run-redact",
        steps=[
            _step_with_diagnostics(
                {
                    "lkw.search_summary.v1": {
                        "num_results": 1,
                        "evidence_count": 1,
                        "query_text": "secret query",
                        "text": "raw chunk",
                        "source_refs": ["docs/a.md"],
                    }
                }
            )
        ],
    )
    evidence = build_lkw_evidence_slice(
        trace,
        capability="local.workspace.search",
        agent_id="local_search",
        run_id="run-redact",
    )
    payload = evidence.diagnostics["lkw.search_summary.v1"]
    assert payload["num_results"] == 1
    assert "query_text" not in payload
    assert "text" not in payload
    assert evidence.schema_version == LKW_EVIDENCE_SCHEMA_VERSION


@pytest.mark.unit
def test_build_lkw_evidence_slice_from_step_diagnostics() -> None:
    evidence = build_lkw_evidence_slice_from_step_diagnostics(
        {
            "lkw.index_summary.v1": {"accepted_count": 1, "ingested_count": 1},
            "ignored.v1": {"x": 1},
        },
        capability="local.workspace.index",
        agent_id="local_indexer",
        run_id="run-index",
        task_id="task-index",
        terminal_status=AgentRunStatus.SUCCEEDED.value,
    )
    assert evidence.task_id == "task-index"
    assert evidence.terminal_status == AgentRunStatus.SUCCEEDED.value
    assert "lkw.index_summary.v1" in evidence.diagnostics
    assert "ignored.v1" not in evidence.diagnostics


@pytest.mark.unit
def test_attach_lkw_evidence_metadata_from_task_result() -> None:
    task_result = TaskResult(
        task_id="task-1",
        run_id="run-1",
        state=TaskState.COMPLETED,
        agent_id="local_search",
        execution_result=AgentExecutionResult(
            agent_id="local_search",
            run_id="run-1",
            status=AgentExecutionStatus.COMPLETED,
            structured_data={
                AcpStructuredDataKey.TRACE_SUMMARY: {
                    "total_steps": 1,
                    "step_diagnostics": {
                        "lkw.search_summary.v1": {
                            "num_results": 2,
                            "evidence_count": 2,
                            "source_refs": ["docs/a.md"],
                        }
                    },
                }
            },
        ),
        metadata={
            TaskResultMetadataKey.APPLICATION_RUN_SUMMARY: {
                "schema_version": "application_run_summary.v1",
                "terminal_status": AgentRunStatus.SUCCEEDED.value,
            }
        },
    )
    metadata: dict[str, object] = dict(task_result.metadata)
    attach_lkw_evidence_metadata(
        metadata,
        task_result=task_result,
        capability="local.workspace.search",
    )
    assert TaskResultMetadataKey.APPLICATION_RUN_SUMMARY in metadata
    evidence = metadata["lkw_evidence.v1"]
    assert isinstance(evidence, dict)
    assert evidence["schema_version"] == "lkw_evidence.v1"
    assert evidence["capability"] == "local.workspace.search"
    assert evidence["agent_id"] == "local_search"
    assert "lkw.search_summary.v1" in evidence["diagnostics"]
