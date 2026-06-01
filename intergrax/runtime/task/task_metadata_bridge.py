# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bridge between typed task contracts and legacy flat metadata dicts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

logger = logging.getLogger(__name__)

from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.context_assembly import (
    CONTEXT_ASSEMBLY_LEGACY_METADATA_KEYS,
    context_assembly_options_from_metadata,
    sync_context_assembly_metadata,
)
from intergrax.contracts.execution_interrupt import ExecutionInterrupt
from intergrax.runtime.human.response_parser import parse_human_response
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.sandbox.sandbox_runtime import SandboxMetadataKey
from intergrax.runtime.task.task_contract import (
    TASK_CONTRACT_METADATA_KEY,
    VERDICT_APPROVE,
    VERDICT_ESCALATE,
    VERDICT_REJECT,
    EscalationStep,
    TaskClassificationState,
    TaskContractPayload,
    TaskExecutionMetrics,
    TaskExecutionOptions,
    TaskGovernanceOptions,
    TaskGovernanceState,
    TaskHumanInput,
    TaskIsolationOptions,
    TaskIsolationSummary,
    TaskLongRunningOptions,
    TaskOrchestrationState,
    TaskOrchestrationSummary,
    TaskPauseRecord,
    TaskResultSummary,
    TaskRetryRecord,
    TaskRuntimeState,
    TaskValidationSummary,
)
from intergrax.runtime.task.task_metadata_keys import (
    TASK_METADATA_LEGACY_OPTION_KEYS,
    TASK_METADATA_LEGACY_RUNTIME_KEYS,
    TaskMetadataKey,
    TaskOrchestrationMetadataKey,
    TaskResultMetadataKey,
)
from intergrax.runtime.workspace.shadow_workspace import ShadowWorkspaceMetadataKey

if TYPE_CHECKING:
    from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
    from intergrax.runtime.task.task import Task, TaskResult

_LEGACY_OPTION_KEYS = frozenset(
    {
        TASK_CONTRACT_METADATA_KEY,
        ShadowWorkspaceMetadataKey.SHADOW_WORKSPACE,
        ShadowWorkspaceMetadataKey.SHADOW_WORKSPACE_CLEANUP,
        SandboxMetadataKey.SANDBOX,
        SandboxMetadataKey.SANDBOX_CLEANUP,
        *TASK_METADATA_LEGACY_OPTION_KEYS,
        *CONTEXT_ASSEMBLY_LEGACY_METADATA_KEYS,
    }
)

_LEGACY_RUNTIME_KEYS = frozenset(
    {
        TASK_CONTRACT_METADATA_KEY,
        *TASK_METADATA_LEGACY_RUNTIME_KEYS,
    }
)


def _has_legacy_keys(metadata: Dict[str, Any], keys: frozenset[str]) -> bool:
    return any(key in metadata for key in keys)


def _truthy(value: Any) -> bool:
    return bool(value)


def _verdict_from_legacy(metadata: Dict[str, Any]) -> Optional[HumanResponseVerdict]:
    raw = metadata.get(TaskMetadataKey.HUMAN_DECISION)
    if raw:
        try:
            return HumanResponseVerdict(str(raw))
        except ValueError:
            return HumanResponseVerdict.UNKNOWN
    if _truthy(metadata.get(TaskMetadataKey.HUMAN_APPROVED)):
        return HumanResponseVerdict.APPROVE
    if _truthy(metadata.get(TaskMetadataKey.HUMAN_REJECTED)):
        return HumanResponseVerdict.REJECT
    if _truthy(metadata.get(TaskMetadataKey.HUMAN_ESCALATED)):
        return HumanResponseVerdict.ESCALATE
    return None


def _verdict_to_contract_value(verdict: HumanResponseVerdict) -> str:
    return verdict.value


def execution_options_from_metadata(metadata: Dict[str, Any]) -> TaskExecutionOptions:
    """Parse intake options from flat metadata (legacy keys or embedded contract)."""
    embedded = metadata.get(TASK_CONTRACT_METADATA_KEY)
    if isinstance(embedded, dict) and "options" in embedded:
        return TaskExecutionOptions.model_validate(embedded["options"])

    isolation = TaskIsolationOptions(
        shadow_workspace=_truthy(metadata.get(ShadowWorkspaceMetadataKey.SHADOW_WORKSPACE)),
        shadow_workspace_cleanup=_truthy(
            metadata.get(ShadowWorkspaceMetadataKey.SHADOW_WORKSPACE_CLEANUP)
        ),
        sandbox=_truthy(metadata.get(SandboxMetadataKey.SANDBOX)),
        sandbox_cleanup=_truthy(metadata.get(SandboxMetadataKey.SANDBOX_CLEANUP)),
    )
    response_text = metadata.get(TaskMetadataKey.HUMAN_RESPONSE)
    verdict = _verdict_from_legacy(metadata)
    if response_text and verdict is None:
        verdict = parse_human_response(str(response_text))

    governance = TaskGovernanceOptions(
        require_human_approval=_truthy(metadata.get(TaskMetadataKey.REQUIRE_HUMAN_APPROVAL)),
        require_human_on_critical=(
            metadata.get(TaskMetadataKey.REQUIRE_HUMAN_ON_CRITICAL, True) is not False
        ),
        high_risk=_truthy(metadata.get(TaskMetadataKey.HIGH_RISK)),
    )
    long_running = TaskLongRunningOptions(
        enabled=_truthy(metadata.get(TaskMetadataKey.LONG_RUNNING)),
        notify_channel=metadata.get(TaskMetadataKey.LONG_RUNNING_NOTIFY_CHANNEL),
        checkpoint_on_pause=(
            metadata.get(TaskMetadataKey.LONG_RUNNING_CHECKPOINT_ON_PAUSE, True) is not False
        ),
        resume_token=(
            str(metadata[TaskMetadataKey.RESUME_TOKEN])
            if metadata.get(TaskMetadataKey.RESUME_TOKEN)
            else None
        ),
    )
    if long_running.resume_token and not long_running.enabled:
        long_running.enabled = True

    context = context_assembly_options_from_metadata(metadata)

    return TaskExecutionOptions(
        isolation=isolation,
        human=TaskHumanInput(
            response_text=str(response_text) if response_text is not None else None,
            verdict=_verdict_to_contract_value(verdict) if verdict is not None else None,
        ),
        governance=governance,
        long_running=long_running,
        context=context,
    )


def execution_options_for_request(request: RuntimeRequest) -> TaskExecutionOptions:
    """Resolve execution options from a ``RuntimeRequest`` (typed contract first)."""
    metadata = request.metadata
    return execution_options_from_metadata(metadata)


def runtime_state_from_metadata(metadata: Dict[str, Any]) -> TaskRuntimeState:
    embedded = metadata.get(TASK_CONTRACT_METADATA_KEY)
    if isinstance(embedded, dict) and "runtime" in embedded:
        return TaskRuntimeState.model_validate(embedded["runtime"])

    human_request = metadata.get(TaskMetadataKey.GOVERNANCE_HUMAN_REQUEST)
    interrupt = metadata.get(TaskMetadataKey.GOVERNANCE_INTERRUPT)
    pause_record = metadata.get(TaskMetadataKey.GOVERNANCE_PAUSE_RECORD)
    chain_raw = metadata.get(TaskMetadataKey.ESCALATION_CHAIN) or []

    governance = TaskGovernanceState(
        paused=_truthy(metadata.get(TaskMetadataKey.GOVERNANCE_PAUSE)),
        human_request=HumanRequest.model_validate(human_request) if human_request else None,
        human_request_created_at=metadata.get(TaskMetadataKey.HUMAN_REQUEST_CREATED_AT),
        human_request_expires_at=metadata.get(TaskMetadataKey.HUMAN_REQUEST_EXPIRES_AT),
        execution_interrupt=(
            ExecutionInterrupt.model_validate(interrupt) if interrupt else None
        ),
        pause_record=TaskPauseRecord.model_validate(pause_record) if pause_record else None,
        escalation_level=int(metadata.get(TaskMetadataKey.ESCALATION_LEVEL, 0)),
        escalation_target=(
            str(metadata[TaskMetadataKey.ESCALATION_TARGET])
            if metadata.get(TaskMetadataKey.ESCALATION_TARGET)
            else None
        ),
        escalation_chain=[
            EscalationStep.model_validate(step) if isinstance(step, dict) else step
            for step in chain_raw
        ],
    )
    classification = TaskClassificationState(
        value=metadata.get(TaskOrchestrationMetadataKey.CLASSIFICATION),
        requested_capability=metadata.get(TaskOrchestrationMetadataKey.REQUESTED_CAPABILITY),
        unsupported_reason=metadata.get(TaskOrchestrationMetadataKey.UNSUPPORTED_REASON),
        risk_level=metadata.get(TaskOrchestrationMetadataKey.RISK_LEVEL),
    )
    orchestration = TaskOrchestrationState(
        plan_id=metadata.get(TaskOrchestrationMetadataKey.PLAN_ID),
        graph_id=metadata.get(TaskOrchestrationMetadataKey.GRAPH_ID),
        needs_more_information=_truthy(
            metadata.get(TaskOrchestrationMetadataKey.NEEDS_MORE_INFORMATION)
        ),
        checkpoint_id=metadata.get(TaskMetadataKey.CHECKPOINT_ID),
        resume_token=metadata.get(TaskMetadataKey.RESUME_TOKEN),
        progress_message=str(metadata.get(TaskMetadataKey.PROGRESS_MESSAGE) or ""),
    )
    return TaskRuntimeState(classification=classification, orchestration=orchestration, governance=governance)


def _warn_legacy_metadata_keys(task: Task) -> None:
    if TASK_CONTRACT_METADATA_KEY in task.metadata:
        return
    legacy_hits = [k for k in task.metadata if k in _LEGACY_OPTION_KEYS]
    if legacy_hits:
        logger.warning(
            "task %s uses legacy flat metadata keys %s; prefer typed Task.options "
            "(Phase Q-X.2)",
            task.task_id,
            legacy_hits[:5],
        )


def hydrate_task_from_metadata(task: Task) -> None:
    _warn_legacy_metadata_keys(task)
    """Merge legacy flat metadata into typed task fields when present."""
    from intergrax.runtime.task.task import Task

    assert isinstance(task, Task)
    meta = task.metadata

    embedded = meta.get(TASK_CONTRACT_METADATA_KEY)
    if isinstance(embedded, dict):
        payload = TaskContractPayload.model_validate(embedded)
        task.options = payload.options
        task.runtime = payload.runtime
        return

    if _has_legacy_keys(meta, _LEGACY_OPTION_KEYS):
        task.options = execution_options_from_metadata(meta)
    if _has_legacy_keys(meta, _LEGACY_RUNTIME_KEYS):
        task.runtime = runtime_state_from_metadata(meta)


def sync_task_metadata(task: Task) -> None:
    """Write typed task fields back to flat metadata for API / JSON compatibility."""
    from intergrax.runtime.task.task import Task

    assert isinstance(task, Task)
    meta = dict(task.metadata)
    opts = task.options
    iso = opts.isolation
    gov_opts = opts.governance
    human = opts.human
    runtime = task.runtime
    cls = runtime.classification
    orch = runtime.orchestration
    gov = runtime.governance

    meta[ShadowWorkspaceMetadataKey.SHADOW_WORKSPACE] = iso.shadow_workspace
    meta[ShadowWorkspaceMetadataKey.SHADOW_WORKSPACE_CLEANUP] = iso.shadow_workspace_cleanup
    meta[SandboxMetadataKey.SANDBOX] = iso.sandbox
    meta[SandboxMetadataKey.SANDBOX_CLEANUP] = iso.sandbox_cleanup
    meta[TaskMetadataKey.REQUIRE_HUMAN_APPROVAL] = gov_opts.require_human_approval
    meta[TaskMetadataKey.REQUIRE_HUMAN_ON_CRITICAL] = gov_opts.require_human_on_critical
    meta[TaskMetadataKey.HIGH_RISK] = gov_opts.high_risk

    if human.response_text is not None:
        meta[TaskMetadataKey.HUMAN_RESPONSE] = human.response_text
    else:
        meta.pop(TaskMetadataKey.HUMAN_RESPONSE, None)

    if human.verdict is not None:
        meta[TaskMetadataKey.HUMAN_DECISION] = human.verdict
        meta[TaskMetadataKey.HUMAN_APPROVED] = human.verdict == VERDICT_APPROVE
        meta[TaskMetadataKey.HUMAN_REJECTED] = human.verdict == VERDICT_REJECT
        meta[TaskMetadataKey.HUMAN_ESCALATED] = human.verdict == VERDICT_ESCALATE
    else:
        meta.pop(TaskMetadataKey.HUMAN_DECISION, None)
        meta.pop(TaskMetadataKey.HUMAN_APPROVED, None)
        meta.pop(TaskMetadataKey.HUMAN_REJECTED, None)
        meta.pop(TaskMetadataKey.HUMAN_ESCALATED, None)

    lr = opts.long_running
    meta[TaskMetadataKey.LONG_RUNNING] = lr.enabled
    if lr.notify_channel is not None:
        meta[TaskMetadataKey.LONG_RUNNING_NOTIFY_CHANNEL] = lr.notify_channel
    else:
        meta.pop(TaskMetadataKey.LONG_RUNNING_NOTIFY_CHANNEL, None)
    meta[TaskMetadataKey.LONG_RUNNING_CHECKPOINT_ON_PAUSE] = lr.checkpoint_on_pause
    if lr.resume_token is not None:
        meta[TaskMetadataKey.RESUME_TOKEN] = lr.resume_token
    elif TaskMetadataKey.RESUME_TOKEN in meta and orch.resume_token is None:
        meta.pop(TaskMetadataKey.RESUME_TOKEN, None)

    sync_context_assembly_metadata(meta, opts.context)

    if orch.checkpoint_id is not None:
        meta[TaskMetadataKey.CHECKPOINT_ID] = orch.checkpoint_id
    if orch.resume_token is not None:
        meta[TaskMetadataKey.RESUME_TOKEN] = orch.resume_token
    if orch.progress_message:
        meta[TaskMetadataKey.PROGRESS_MESSAGE] = orch.progress_message
    else:
        meta.pop(TaskMetadataKey.PROGRESS_MESSAGE, None)

    if cls.value is not None:
        meta[TaskOrchestrationMetadataKey.CLASSIFICATION] = cls.value
    if cls.requested_capability is not None:
        meta[TaskOrchestrationMetadataKey.REQUESTED_CAPABILITY] = cls.requested_capability
    if cls.unsupported_reason is not None:
        meta[TaskOrchestrationMetadataKey.UNSUPPORTED_REASON] = cls.unsupported_reason
    if cls.risk_level is not None:
        meta[TaskOrchestrationMetadataKey.RISK_LEVEL] = cls.risk_level

    if orch.plan_id is not None:
        meta[TaskOrchestrationMetadataKey.PLAN_ID] = orch.plan_id
    if orch.graph_id is not None:
        meta[TaskOrchestrationMetadataKey.GRAPH_ID] = orch.graph_id
    meta[TaskOrchestrationMetadataKey.NEEDS_MORE_INFORMATION] = orch.needs_more_information

    meta[TaskMetadataKey.GOVERNANCE_PAUSE] = gov.paused
    if gov.human_request is not None:
        meta[TaskMetadataKey.GOVERNANCE_HUMAN_REQUEST] = gov.human_request.model_dump()
    else:
        meta.pop(TaskMetadataKey.GOVERNANCE_HUMAN_REQUEST, None)
    if gov.human_request_created_at is not None:
        meta[TaskMetadataKey.HUMAN_REQUEST_CREATED_AT] = gov.human_request_created_at
    else:
        meta.pop(TaskMetadataKey.HUMAN_REQUEST_CREATED_AT, None)
    if gov.human_request_expires_at is not None:
        meta[TaskMetadataKey.HUMAN_REQUEST_EXPIRES_AT] = gov.human_request_expires_at
    else:
        meta.pop(TaskMetadataKey.HUMAN_REQUEST_EXPIRES_AT, None)
    if gov.execution_interrupt is not None:
        meta[TaskMetadataKey.GOVERNANCE_INTERRUPT] = gov.execution_interrupt.model_dump()
    else:
        meta.pop(TaskMetadataKey.GOVERNANCE_INTERRUPT, None)
    if gov.pause_record is not None:
        meta[TaskMetadataKey.GOVERNANCE_PAUSE_RECORD] = gov.pause_record.model_dump()
    else:
        meta.pop(TaskMetadataKey.GOVERNANCE_PAUSE_RECORD, None)

    meta[TaskMetadataKey.ESCALATION_LEVEL] = gov.escalation_level
    if gov.escalation_target is not None:
        meta[TaskMetadataKey.ESCALATION_TARGET] = gov.escalation_target
    else:
        meta.pop(TaskMetadataKey.ESCALATION_TARGET, None)
    if gov.escalation_chain:
        meta[TaskMetadataKey.ESCALATION_CHAIN] = [step.model_dump() for step in gov.escalation_chain]
    else:
        meta.pop(TaskMetadataKey.ESCALATION_CHAIN, None)

    meta[TASK_CONTRACT_METADATA_KEY] = TaskContractPayload(
        options=opts,
        runtime=runtime,
    ).model_dump(mode="json")

    task.metadata = meta


def task_to_request_metadata(task: Task) -> Dict[str, Any]:
    sync_task_metadata(task)
    return dict(task.metadata)


def result_summary_from_metadata(metadata: Dict[str, Any]) -> TaskResultSummary:
    embedded = metadata.get(TaskResultMetadataKey.TASK_RESULT)
    if isinstance(embedded, dict):
        return TaskResultSummary.model_validate(embedded)

    retries_raw = metadata.get(TaskResultMetadataKey.RETRIES) or []
    chain_raw = metadata.get(TaskMetadataKey.ESCALATION_CHAIN) or []
    gov_req = metadata.get(TaskMetadataKey.GOVERNANCE_HUMAN_REQUEST)

    return TaskResultSummary(
        validation=TaskValidationSummary(
            valid=_truthy(metadata.get(TaskResultMetadataKey.VALIDATION_VALID)),
            errors=list(metadata.get(TaskResultMetadataKey.VALIDATION_ERRORS) or []),
            warnings=list(metadata.get(TaskResultMetadataKey.VALIDATION_WARNINGS) or []),
        ),
        metrics=TaskExecutionMetrics(
            cost=float(metadata.get(TaskResultMetadataKey.EXECUTION_COST) or 0.0),
            total_tokens=int(metadata.get(TaskResultMetadataKey.EXECUTION_TOTAL_TOKENS) or 0),
            runtime_events=int(metadata.get(TaskResultMetadataKey.RUNTIME_EVENTS) or 0),
            task_trace_events=int(metadata.get(TaskResultMetadataKey.TASK_TRACE_EVENTS) or 0),
        ),
        isolation=TaskIsolationSummary(
            shadow_workspace_id=metadata.get(TaskResultMetadataKey.SHADOW_WORKSPACE_ID),
            shadow_artifact_count=metadata.get(TaskResultMetadataKey.SHADOW_ARTIFACT_COUNT),
            sandbox_session_id=metadata.get(TaskResultMetadataKey.SANDBOX_SESSION_ID),
            sandbox_operation_count=metadata.get(TaskResultMetadataKey.SANDBOX_OPERATION_COUNT),
        ),
        orchestration=TaskOrchestrationSummary(
            classification=str(metadata.get(TaskOrchestrationMetadataKey.CLASSIFICATION) or ""),
            plan_id=str(metadata.get(TaskOrchestrationMetadataKey.PLAN_ID) or ""),
            graph_id=str(metadata.get(TaskOrchestrationMetadataKey.GRAPH_ID) or ""),
            graph_node_count=int(metadata.get(TaskResultMetadataKey.GRAPH_NODE_COUNT) or 0),
            agent_count=int(metadata.get(TaskResultMetadataKey.AGENT_COUNT) or 0),
            agent_ids=list(metadata.get(TaskResultMetadataKey.AGENT_IDS) or []),
            retry_count=int(metadata.get(TaskResultMetadataKey.RETRY_COUNT) or 0),
            retries=[
                TaskRetryRecord.model_validate(r) if isinstance(r, dict) else r
                for r in retries_raw
            ],
            all_completed=_truthy(metadata.get(TaskResultMetadataKey.ALL_COMPLETED)),
        ),
        escalation_level=int(metadata.get(TaskMetadataKey.ESCALATION_LEVEL) or 0),
        escalation_chain=[
            EscalationStep.model_validate(step) if isinstance(step, dict) else step
            for step in chain_raw
        ],
        governance_human_request=gov_req if isinstance(gov_req, dict) else None,
        checkpoint_id=metadata.get(TaskMetadataKey.CHECKPOINT_ID),
        resume_token=metadata.get(TaskMetadataKey.RESUME_TOKEN),
        progress_message=str(metadata.get(TaskMetadataKey.PROGRESS_MESSAGE) or ""),
    )


def sync_result_metadata(result: TaskResult) -> None:
    from intergrax.runtime.task.task import TaskResult

    assert isinstance(result, TaskResult)
    summary = result.summary
    meta = dict(result.metadata)

    meta[TaskResultMetadataKey.VALIDATION_VALID] = summary.validation.valid
    meta[TaskResultMetadataKey.VALIDATION_ERRORS] = list(summary.validation.errors)
    meta[TaskResultMetadataKey.VALIDATION_WARNINGS] = list(summary.validation.warnings)
    meta[TaskResultMetadataKey.EXECUTION_COST] = summary.metrics.cost
    meta[TaskResultMetadataKey.EXECUTION_TOTAL_TOKENS] = summary.metrics.total_tokens
    meta[TaskResultMetadataKey.RUNTIME_EVENTS] = summary.metrics.runtime_events
    meta[TaskResultMetadataKey.TASK_TRACE_EVENTS] = summary.metrics.task_trace_events

    orch = summary.orchestration
    meta[TaskOrchestrationMetadataKey.CLASSIFICATION] = orch.classification
    meta[TaskOrchestrationMetadataKey.PLAN_ID] = orch.plan_id
    meta[TaskOrchestrationMetadataKey.GRAPH_ID] = orch.graph_id
    meta[TaskResultMetadataKey.GRAPH_NODE_COUNT] = orch.graph_node_count
    meta[TaskResultMetadataKey.AGENT_COUNT] = orch.agent_count
    meta[TaskResultMetadataKey.AGENT_IDS] = list(orch.agent_ids)
    meta[TaskResultMetadataKey.RETRY_COUNT] = orch.retry_count
    meta[TaskResultMetadataKey.RETRIES] = [r.model_dump() for r in orch.retries]
    meta[TaskResultMetadataKey.ALL_COMPLETED] = orch.all_completed

    iso = summary.isolation
    if iso.shadow_workspace_id is not None:
        meta[TaskResultMetadataKey.SHADOW_WORKSPACE_ID] = iso.shadow_workspace_id
    if iso.shadow_artifact_count is not None:
        meta[TaskResultMetadataKey.SHADOW_ARTIFACT_COUNT] = iso.shadow_artifact_count
    if iso.sandbox_session_id is not None:
        meta[TaskResultMetadataKey.SANDBOX_SESSION_ID] = iso.sandbox_session_id
    if iso.sandbox_operation_count is not None:
        meta[TaskResultMetadataKey.SANDBOX_OPERATION_COUNT] = iso.sandbox_operation_count

    if summary.escalation_level > 0:
        meta[TaskMetadataKey.ESCALATION_LEVEL] = summary.escalation_level
    elif TaskMetadataKey.ESCALATION_LEVEL in meta and summary.escalation_level == 0:
        meta.pop(TaskMetadataKey.ESCALATION_LEVEL, None)
    if summary.escalation_chain:
        meta[TaskMetadataKey.ESCALATION_CHAIN] = [
            step.model_dump() for step in summary.escalation_chain
        ]
    if summary.governance_human_request is not None:
        meta[TaskMetadataKey.GOVERNANCE_HUMAN_REQUEST] = summary.governance_human_request

    if summary.checkpoint_id is not None:
        meta[TaskMetadataKey.CHECKPOINT_ID] = summary.checkpoint_id
    if summary.resume_token is not None:
        meta[TaskMetadataKey.RESUME_TOKEN] = summary.resume_token
    if summary.progress_message:
        meta[TaskMetadataKey.PROGRESS_MESSAGE] = summary.progress_message

    meta[TaskResultMetadataKey.TASK_RESULT] = summary.model_dump(mode="json")
    result.metadata = meta


def hydrate_result_from_metadata(result: TaskResult) -> None:
    from intergrax.runtime.task.task import TaskResult

    assert isinstance(result, TaskResult)
    result.summary = result_summary_from_metadata(result.metadata)

