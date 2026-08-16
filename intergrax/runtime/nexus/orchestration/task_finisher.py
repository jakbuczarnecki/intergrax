# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Nexus task result assembly (Phase Q-N.1 decomposition)."""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.runtime_cost import aggregate_execution_metrics
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.human.request_contract import human_request_event_payload
from intergrax.runtime.nexus.orchestration.run_artifact_bundle_builder import build_run_artifact_bundle
from intergrax.contracts.run_artifact_bundle import RUN_ARTIFACT_BUNDLE_METADATA_KEY
from intergrax.runtime.nexus.orchestration.application_run_summary_builder import (
    build_application_run_summary,
)
from intergrax.runtime.nexus.orchestration.workspace_cleanup import (
    cleanup_sandbox_for_task,
    cleanup_shadow_for_task,
    clear_isolation_refs_in_task_env_state,
)
from intergrax.runtime.nexus.planning.task_planner import NexusPlan
from intergrax.runtime.nexus.response.final_response_composer import FinalResponseComposer
from intergrax.runtime.nexus.retry.retry_engine import RetryRecord
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.sandbox.sandbox_runtime import SANDBOX_SESSION_ID_KEY
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.runtime.task.task_contract import (
    TaskExecutionMetrics,
    TaskIsolationSummary,
    TaskOrchestrationSummary,
    TaskResultSummary,
    TaskRetryRecord,
    TaskValidationSummary,
)
from intergrax.runtime.task.task_metadata_keys import TaskMetadataKey, TaskResultMetadataKey
from intergrax.runtime.task.task_trace import TaskTraceEmitter
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.runtime.workspace.shadow_workspace import SHADOW_WORKSPACE_ID_KEY
from intergrax.utils.time_provider import SystemTimeProvider

if TYPE_CHECKING:
    from intergrax.contracts.execution_identity import RunId


def build_nexus_task_result(
    task: Task,
    trace_emitter: TaskTraceEmitter,
    *,
    answer: str,
    executions: List[AgentExecutionResult],
    validation: ValidationResult,
    plan: Optional[NexusPlan],
    retry_records: List[RetryRecord],
    graph_id: str,
    composer: FinalResponseComposer,
    event_bus: RuntimeEventBus,
    shadow_manager: ShadowWorkspaceManager,
    sandbox_manager: SandboxSessionManager,
    run_id: Optional["RunId"] = None,
) -> TaskResult:
    primary = executions[-1] if executions else None
    composer_meta = composer.compose_metadata(
        executions,
        classification=task.classification or "",
        plan_id=plan.plan_id if plan else "",
        retry_count=len(retry_records),
    )
    execution_metrics = aggregate_execution_metrics(executions)

    gov_human_request = None
    gov = task.runtime.governance
    if gov.human_request is not None:
        gov_human_request = human_request_event_payload(
            gov.human_request,
            created_at_utc=gov.human_request_created_at,
            expires_at_utc=gov.human_request_expires_at,
        )
    elif executions and executions[-1].human_request:
        gov_human_request = human_request_event_payload(
            executions[-1].human_request,
            created_at_utc=SystemTimeProvider.utc_now().isoformat(),
        )

    isolation = TaskIsolationSummary()
    if primary and primary.structured_data.get(SHADOW_WORKSPACE_ID_KEY):
        isolation.shadow_workspace_id = str(primary.structured_data[SHADOW_WORKSPACE_ID_KEY])
        artifact_count = primary.structured_data.get("shadow_artifact_count")
        if artifact_count is not None:
            isolation.shadow_artifact_count = int(artifact_count)

    if primary and primary.structured_data.get(SANDBOX_SESSION_ID_KEY):
        isolation.sandbox_session_id = str(primary.structured_data[SANDBOX_SESSION_ID_KEY])
        operation_count = primary.structured_data.get("sandbox_operation_count")
        if operation_count is not None:
            isolation.sandbox_operation_count = int(operation_count)

    summary = TaskResultSummary(
        validation=TaskValidationSummary(
            valid=validation.valid,
            errors=list(validation.errors),
            warnings=list(validation.warnings),
        ),
        metrics=TaskExecutionMetrics(
            cost=execution_metrics.cost,
            total_tokens=execution_metrics.total_tokens,
            runtime_events=len(event_bus.history),
            task_trace_events=len(trace_emitter.events),
        ),
        isolation=isolation,
        orchestration=TaskOrchestrationSummary(
            classification=composer_meta.get("classification", ""),
            plan_id=composer_meta.get("plan_id", ""),
            graph_id=graph_id,
            graph_node_count=len(plan.steps) if plan else 0,
            agent_count=composer_meta.get("agent_count", 0),
            agent_ids=list(composer_meta.get("agent_ids") or []),
            retry_count=composer_meta.get("retry_count", 0),
            retries=[
                TaskRetryRecord(
                    attempt=r.attempt,
                    agent_id=r.agent_id,
                    alternate_agent_id=r.alternate_agent_id,
                    reason=r.reason,
                )
                for r in retry_records
            ],
            all_completed=bool(composer_meta.get("all_completed")),
        ),
        escalation_level=HumanPauseCoordinator.escalation_level(task),
        escalation_chain=list(task.runtime.governance.escalation_chain),
        governance_human_request=gov_human_request,
        checkpoint_id=task.runtime.orchestration.checkpoint_id,
        resume_token=task.runtime.orchestration.resume_token,
        progress_message=task.runtime.orchestration.progress_message,
    )

    artifact_bundle = build_run_artifact_bundle(
        task=task,
        graph_id=graph_id,
        executions=executions,
        shadow_manager=shadow_manager,
        sandbox_manager=sandbox_manager,
    )
    bundle_payload = artifact_bundle.model_dump(mode="json")

    cleanup_shadow_for_task(task, executions, shadow_manager=shadow_manager)
    cleanup_sandbox_for_task(task, executions, sandbox_manager=sandbox_manager)
    clear_isolation_refs_in_task_env_state(task)

    task.sync_metadata()
    result = TaskResult(
        task_id=task.task_id,
        run_id=run_id or (primary.run_id if primary else None),
        state=task.state,
        answer=answer,
        agent_id=primary.agent_id if primary else task.agent_id,
        execution_result=primary,
        summary=summary,
        metadata=dict(composer_meta),
    )
    for key in (
        TaskMetadataKey.GOVERNANCE_HUMAN_REQUEST,
        TaskMetadataKey.HUMAN_REQUEST_CREATED_AT,
        TaskMetadataKey.HUMAN_REQUEST_EXPIRES_AT,
    ):
        if key in task.metadata:
            result.metadata[key] = task.metadata[key]
    if plan and plan.plan_metadata:
        result.metadata.update(plan.plan_metadata)
    app_summary = build_application_run_summary(
        task_id=task.task_id,
        graph_id=graph_id,
        executions=executions,
    )
    app_summary.metadata[RUN_ARTIFACT_BUNDLE_METADATA_KEY] = bundle_payload
    result.metadata[TaskResultMetadataKey.APPLICATION_RUN_SUMMARY] = app_summary.model_dump(
        mode="json"
    )
    result.metadata[TaskResultMetadataKey.RUN_ARTIFACT_BUNDLE] = bundle_payload
    result.sync_metadata()
    return result
