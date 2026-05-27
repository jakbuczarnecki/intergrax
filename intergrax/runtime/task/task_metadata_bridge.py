# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bridge between typed task contracts and legacy flat metadata dicts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.execution_interrupt import ExecutionInterrupt
from intergrax.runtime.human.response_parser import parse_human_response
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.sandbox.sandbox_runtime import (
    SANDBOX_CLEANUP_KEY,
    SANDBOX_FLAG,
)
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
    CHECKPOINT_ID_KEY,
    ESCALATION_CHAIN_KEY,
    ESCALATION_LEVEL_KEY,
    ESCALATION_TARGET_KEY,
    GOVERNANCE_HUMAN_REQUEST_KEY,
    GOVERNANCE_INTERRUPT_KEY,
    GOVERNANCE_PAUSE_KEY,
    GOVERNANCE_PAUSE_RECORD_KEY,
    HUMAN_APPROVED_KEY,
    HUMAN_DECISION_KEY,
    HUMAN_ESCALATED_KEY,
    HUMAN_REJECTED_KEY,
    HUMAN_RESPONSE_KEY,
    LONG_RUNNING_CHECKPOINT_ON_PAUSE_KEY,
    LONG_RUNNING_FLAG,
    LONG_RUNNING_NOTIFY_CHANNEL_KEY,
    PROGRESS_MESSAGE_KEY,
    RESUME_TOKEN_KEY,
)


from intergrax.runtime.workspace.shadow_workspace import (
    SHADOW_WORKSPACE_CLEANUP_KEY,
    SHADOW_WORKSPACE_FLAG,
)

_LEGACY_OPTION_KEYS = frozenset(
    {
        TASK_CONTRACT_METADATA_KEY,
        SHADOW_WORKSPACE_FLAG,
        SHADOW_WORKSPACE_CLEANUP_KEY,
        SANDBOX_FLAG,
        SANDBOX_CLEANUP_KEY,
        HUMAN_RESPONSE_KEY,
        HUMAN_DECISION_KEY,
        HUMAN_APPROVED_KEY,
        HUMAN_REJECTED_KEY,
        HUMAN_ESCALATED_KEY,
        "require_human_approval",
        "require_human_on_critical",
        "high_risk",
        LONG_RUNNING_FLAG,
        LONG_RUNNING_NOTIFY_CHANNEL_KEY,
        LONG_RUNNING_CHECKPOINT_ON_PAUSE_KEY,
        RESUME_TOKEN_KEY,
    }
)

_LEGACY_RUNTIME_KEYS = frozenset(
    {
        TASK_CONTRACT_METADATA_KEY,
        "classification",
        "requested_capability",
        "unsupported_reason",
        "risk_level",
        "plan_id",
        "graph_id",
        "needs_more_information",
        GOVERNANCE_PAUSE_KEY,
        GOVERNANCE_HUMAN_REQUEST_KEY,
        GOVERNANCE_INTERRUPT_KEY,
        GOVERNANCE_PAUSE_RECORD_KEY,
        ESCALATION_LEVEL_KEY,
        ESCALATION_TARGET_KEY,
        ESCALATION_CHAIN_KEY,
        CHECKPOINT_ID_KEY,
        PROGRESS_MESSAGE_KEY,
    }
)


def _has_legacy_keys(metadata: Dict[str, Any], keys: frozenset[str]) -> bool:
    return any(key in metadata for key in keys)


def _truthy(value: Any) -> bool:
    return bool(value)


def _verdict_from_legacy(metadata: Dict[str, Any]) -> Optional[HumanResponseVerdict]:
    raw = metadata.get(HUMAN_DECISION_KEY)
    if raw:
        try:
            return HumanResponseVerdict(str(raw))
        except ValueError:
            return HumanResponseVerdict.UNKNOWN
    if _truthy(metadata.get(HUMAN_APPROVED_KEY)):
        return HumanResponseVerdict.APPROVE
    if _truthy(metadata.get(HUMAN_REJECTED_KEY)):
        return HumanResponseVerdict.REJECT
    if _truthy(metadata.get(HUMAN_ESCALATED_KEY)):
        return HumanResponseVerdict.ESCALATE
    return None


def _verdict_to_contract_value(verdict: HumanResponseVerdict) -> str:
    return verdict.value


def execution_options_from_metadata(metadata: Dict[str, Any]) -> TaskExecutionOptions:
    """Parse intake options from RuntimeRequest metadata or legacy task metadata."""
    embedded = metadata.get(TASK_CONTRACT_METADATA_KEY)
    if isinstance(embedded, dict) and "options" in embedded:
        return TaskExecutionOptions.model_validate(embedded["options"])

    isolation = TaskIsolationOptions(
        shadow_workspace=_truthy(metadata.get(SHADOW_WORKSPACE_FLAG)),
        shadow_workspace_cleanup=_truthy(metadata.get(SHADOW_WORKSPACE_CLEANUP_KEY)),
        sandbox=_truthy(metadata.get(SANDBOX_FLAG)),
        sandbox_cleanup=_truthy(metadata.get(SANDBOX_CLEANUP_KEY)),
    )
    response_text = metadata.get(HUMAN_RESPONSE_KEY)
    verdict = _verdict_from_legacy(metadata)
    if response_text and verdict is None:
        verdict = parse_human_response(str(response_text))

    governance = TaskGovernanceOptions(
        require_human_approval=_truthy(metadata.get("require_human_approval")),
        require_human_on_critical=metadata.get("require_human_on_critical", True) is not False,
        high_risk=_truthy(metadata.get("high_risk")),
    )
    long_running = TaskLongRunningOptions(
        enabled=_truthy(metadata.get(LONG_RUNNING_FLAG)),
        notify_channel=metadata.get(LONG_RUNNING_NOTIFY_CHANNEL_KEY),
        checkpoint_on_pause=metadata.get(LONG_RUNNING_CHECKPOINT_ON_PAUSE_KEY, True) is not False,
        resume_token=(
            str(metadata[RESUME_TOKEN_KEY]) if metadata.get(RESUME_TOKEN_KEY) else None
        ),
    )
    if long_running.resume_token and not long_running.enabled:
        long_running.enabled = True
    return TaskExecutionOptions(
        isolation=isolation,
        human=TaskHumanInput(
            response_text=str(response_text) if response_text is not None else None,
            verdict=_verdict_to_contract_value(verdict) if verdict is not None else None,
        ),
        governance=governance,
        long_running=long_running,
    )


def runtime_state_from_metadata(metadata: Dict[str, Any]) -> TaskRuntimeState:
    embedded = metadata.get(TASK_CONTRACT_METADATA_KEY)
    if isinstance(embedded, dict) and "runtime" in embedded:
        return TaskRuntimeState.model_validate(embedded["runtime"])

    human_request = metadata.get(GOVERNANCE_HUMAN_REQUEST_KEY)
    interrupt = metadata.get(GOVERNANCE_INTERRUPT_KEY)
    pause_record = metadata.get(GOVERNANCE_PAUSE_RECORD_KEY)
    chain_raw = metadata.get(ESCALATION_CHAIN_KEY) or []

    governance = TaskGovernanceState(
        paused=_truthy(metadata.get(GOVERNANCE_PAUSE_KEY)),
        human_request=HumanRequest.model_validate(human_request) if human_request else None,
        execution_interrupt=(
            ExecutionInterrupt.model_validate(interrupt) if interrupt else None
        ),
        pause_record=TaskPauseRecord.model_validate(pause_record) if pause_record else None,
        escalation_level=int(metadata.get(ESCALATION_LEVEL_KEY, 0)),
        escalation_target=(
            str(metadata[ESCALATION_TARGET_KEY]) if metadata.get(ESCALATION_TARGET_KEY) else None
        ),
        escalation_chain=[
            EscalationStep.model_validate(step) if isinstance(step, dict) else step
            for step in chain_raw
        ],
    )
    classification = TaskClassificationState(
        value=metadata.get("classification"),
        requested_capability=metadata.get("requested_capability"),
        unsupported_reason=metadata.get("unsupported_reason"),
        risk_level=metadata.get("risk_level"),
    )
    orchestration = TaskOrchestrationState(
        plan_id=metadata.get("plan_id"),
        graph_id=metadata.get("graph_id"),
        needs_more_information=_truthy(metadata.get("needs_more_information")),
        checkpoint_id=metadata.get(CHECKPOINT_ID_KEY),
        resume_token=metadata.get(RESUME_TOKEN_KEY),
        progress_message=str(metadata.get(PROGRESS_MESSAGE_KEY) or ""),
    )
    return TaskRuntimeState(classification=classification, orchestration=orchestration, governance=governance)


def hydrate_task_from_metadata(task: Task) -> None:
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

    meta[SHADOW_WORKSPACE_FLAG] = iso.shadow_workspace
    meta[SHADOW_WORKSPACE_CLEANUP_KEY] = iso.shadow_workspace_cleanup
    meta[SANDBOX_FLAG] = iso.sandbox
    meta[SANDBOX_CLEANUP_KEY] = iso.sandbox_cleanup
    meta["require_human_approval"] = gov_opts.require_human_approval
    meta["require_human_on_critical"] = gov_opts.require_human_on_critical
    meta["high_risk"] = gov_opts.high_risk

    if human.response_text is not None:
        meta[HUMAN_RESPONSE_KEY] = human.response_text
    else:
        meta.pop(HUMAN_RESPONSE_KEY, None)

    if human.verdict is not None:
        meta[HUMAN_DECISION_KEY] = human.verdict
        meta[HUMAN_APPROVED_KEY] = human.verdict == VERDICT_APPROVE
        meta[HUMAN_REJECTED_KEY] = human.verdict == VERDICT_REJECT
        meta[HUMAN_ESCALATED_KEY] = human.verdict == VERDICT_ESCALATE
    else:
        meta.pop(HUMAN_DECISION_KEY, None)
        meta.pop(HUMAN_APPROVED_KEY, None)
        meta.pop(HUMAN_REJECTED_KEY, None)
        meta.pop(HUMAN_ESCALATED_KEY, None)

    lr = opts.long_running
    meta[LONG_RUNNING_FLAG] = lr.enabled
    if lr.notify_channel is not None:
        meta[LONG_RUNNING_NOTIFY_CHANNEL_KEY] = lr.notify_channel
    else:
        meta.pop(LONG_RUNNING_NOTIFY_CHANNEL_KEY, None)
    meta[LONG_RUNNING_CHECKPOINT_ON_PAUSE_KEY] = lr.checkpoint_on_pause
    if lr.resume_token is not None:
        meta[RESUME_TOKEN_KEY] = lr.resume_token
    elif RESUME_TOKEN_KEY in meta and orch.resume_token is None:
        meta.pop(RESUME_TOKEN_KEY, None)

    if orch.checkpoint_id is not None:
        meta[CHECKPOINT_ID_KEY] = orch.checkpoint_id
    if orch.resume_token is not None:
        meta[RESUME_TOKEN_KEY] = orch.resume_token
    if orch.progress_message:
        meta[PROGRESS_MESSAGE_KEY] = orch.progress_message
    else:
        meta.pop(PROGRESS_MESSAGE_KEY, None)

    if cls.value is not None:
        meta["classification"] = cls.value
    if cls.requested_capability is not None:
        meta["requested_capability"] = cls.requested_capability
    if cls.unsupported_reason is not None:
        meta["unsupported_reason"] = cls.unsupported_reason
    if cls.risk_level is not None:
        meta["risk_level"] = cls.risk_level

    if orch.plan_id is not None:
        meta["plan_id"] = orch.plan_id
    if orch.graph_id is not None:
        meta["graph_id"] = orch.graph_id
    meta["needs_more_information"] = orch.needs_more_information

    meta[GOVERNANCE_PAUSE_KEY] = gov.paused
    if gov.human_request is not None:
        meta[GOVERNANCE_HUMAN_REQUEST_KEY] = gov.human_request.model_dump()
    else:
        meta.pop(GOVERNANCE_HUMAN_REQUEST_KEY, None)
    if gov.execution_interrupt is not None:
        meta[GOVERNANCE_INTERRUPT_KEY] = gov.execution_interrupt.model_dump()
    else:
        meta.pop(GOVERNANCE_INTERRUPT_KEY, None)
    if gov.pause_record is not None:
        meta[GOVERNANCE_PAUSE_RECORD_KEY] = gov.pause_record.model_dump()
    else:
        meta.pop(GOVERNANCE_PAUSE_RECORD_KEY, None)

    meta[ESCALATION_LEVEL_KEY] = gov.escalation_level
    if gov.escalation_target is not None:
        meta[ESCALATION_TARGET_KEY] = gov.escalation_target
    else:
        meta.pop(ESCALATION_TARGET_KEY, None)
    if gov.escalation_chain:
        meta[ESCALATION_CHAIN_KEY] = [step.model_dump() for step in gov.escalation_chain]
    else:
        meta.pop(ESCALATION_CHAIN_KEY, None)

    meta[TASK_CONTRACT_METADATA_KEY] = TaskContractPayload(
        options=opts,
        runtime=runtime,
    ).model_dump(mode="json")

    task.metadata = meta


def task_to_request_metadata(task: Task) -> Dict[str, Any]:
    sync_task_metadata(task)
    return dict(task.metadata)


def result_summary_from_metadata(metadata: Dict[str, Any]) -> TaskResultSummary:
    embedded = metadata.get("task_result.v1")
    if isinstance(embedded, dict):
        return TaskResultSummary.model_validate(embedded)

    retries_raw = metadata.get("retries") or []
    chain_raw = metadata.get(ESCALATION_CHAIN_KEY) or []
    gov_req = metadata.get(GOVERNANCE_HUMAN_REQUEST_KEY)

    return TaskResultSummary(
        validation=TaskValidationSummary(
            valid=_truthy(metadata.get("validation_valid")),
            errors=list(metadata.get("validation_errors") or []),
            warnings=list(metadata.get("validation_warnings") or []),
        ),
        metrics=TaskExecutionMetrics(
            cost=float(metadata.get("execution_cost") or 0.0),
            total_tokens=int(metadata.get("execution_total_tokens") or 0),
            runtime_events=int(metadata.get("runtime_events") or 0),
            task_trace_events=int(metadata.get("task_trace_events") or 0),
        ),
        isolation=TaskIsolationSummary(
            shadow_workspace_id=metadata.get("shadow_workspace_id"),
            shadow_artifact_count=metadata.get("shadow_artifact_count"),
            sandbox_session_id=metadata.get("sandbox_session_id"),
            sandbox_operation_count=metadata.get("sandbox_operation_count"),
        ),
        orchestration=TaskOrchestrationSummary(
            classification=str(metadata.get("classification") or ""),
            plan_id=str(metadata.get("plan_id") or ""),
            graph_id=str(metadata.get("graph_id") or ""),
            graph_node_count=int(metadata.get("graph_node_count") or 0),
            agent_count=int(metadata.get("agent_count") or 0),
            agent_ids=list(metadata.get("agent_ids") or []),
            retry_count=int(metadata.get("retry_count") or 0),
            retries=[
                TaskRetryRecord.model_validate(r) if isinstance(r, dict) else r
                for r in retries_raw
            ],
            all_completed=_truthy(metadata.get("all_completed")),
        ),
        escalation_level=int(metadata.get(ESCALATION_LEVEL_KEY) or 0),
        escalation_chain=[
            EscalationStep.model_validate(step) if isinstance(step, dict) else step
            for step in chain_raw
        ],
        governance_human_request=gov_req if isinstance(gov_req, dict) else None,
        checkpoint_id=metadata.get(CHECKPOINT_ID_KEY),
        resume_token=metadata.get(RESUME_TOKEN_KEY),
        progress_message=str(metadata.get(PROGRESS_MESSAGE_KEY) or ""),
    )


def sync_result_metadata(result: TaskResult) -> None:
    from intergrax.runtime.task.task import TaskResult

    assert isinstance(result, TaskResult)
    summary = result.summary
    meta = dict(result.metadata)

    meta["validation_valid"] = summary.validation.valid
    meta["validation_errors"] = list(summary.validation.errors)
    meta["validation_warnings"] = list(summary.validation.warnings)
    meta["execution_cost"] = summary.metrics.cost
    meta["execution_total_tokens"] = summary.metrics.total_tokens
    meta["runtime_events"] = summary.metrics.runtime_events
    meta["task_trace_events"] = summary.metrics.task_trace_events

    orch = summary.orchestration
    meta["classification"] = orch.classification
    meta["plan_id"] = orch.plan_id
    meta["graph_id"] = orch.graph_id
    meta["graph_node_count"] = orch.graph_node_count
    meta["agent_count"] = orch.agent_count
    meta["agent_ids"] = list(orch.agent_ids)
    meta["retry_count"] = orch.retry_count
    meta["retries"] = [r.model_dump() for r in orch.retries]
    meta["all_completed"] = orch.all_completed

    iso = summary.isolation
    if iso.shadow_workspace_id is not None:
        meta["shadow_workspace_id"] = iso.shadow_workspace_id
    if iso.shadow_artifact_count is not None:
        meta["shadow_artifact_count"] = iso.shadow_artifact_count
    if iso.sandbox_session_id is not None:
        meta["sandbox_session_id"] = iso.sandbox_session_id
    if iso.sandbox_operation_count is not None:
        meta["sandbox_operation_count"] = iso.sandbox_operation_count

    if summary.escalation_level > 0:
        meta[ESCALATION_LEVEL_KEY] = summary.escalation_level
    elif ESCALATION_LEVEL_KEY in meta and summary.escalation_level == 0:
        meta.pop(ESCALATION_LEVEL_KEY, None)
    if summary.escalation_chain:
        meta[ESCALATION_CHAIN_KEY] = [step.model_dump() for step in summary.escalation_chain]
    if summary.governance_human_request is not None:
        meta[GOVERNANCE_HUMAN_REQUEST_KEY] = summary.governance_human_request

    if summary.checkpoint_id is not None:
        meta[CHECKPOINT_ID_KEY] = summary.checkpoint_id
    if summary.resume_token is not None:
        meta[RESUME_TOKEN_KEY] = summary.resume_token
    if summary.progress_message:
        meta[PROGRESS_MESSAGE_KEY] = summary.progress_message

    meta["task_result.v1"] = summary.model_dump(mode="json")
    result.metadata = meta


def hydrate_result_from_metadata(result: TaskResult) -> None:
    from intergrax.runtime.task.task import TaskResult

    assert isinstance(result, TaskResult)
    result.summary = result_summary_from_metadata(result.metadata)
