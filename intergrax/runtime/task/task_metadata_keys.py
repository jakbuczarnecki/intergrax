# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Flat task metadata keys for JSON/API bridge (prefer typed ``Task`` fields)."""

from __future__ import annotations

from enum import StrEnum


class TaskMetadataKey(StrEnum):
    """Governance, HITL, escalation, long-running and shared-context flat keys."""

    GOVERNANCE_HUMAN_REQUEST = "governance_human_request"
    GOVERNANCE_INTERRUPT = "governance_interrupt"
    GOVERNANCE_PAUSE = "governance_pause"
    GOVERNANCE_PAUSE_RECORD = "governance_pause_record"

    HUMAN_RESPONSE = "human_response"
    HUMAN_DECISION = "human_decision"
    HUMAN_APPROVED = "human_approved"
    HUMAN_REJECTED = "human_rejected"
    HUMAN_ESCALATED = "human_escalated"
    HUMAN_REQUEST_CREATED_AT = "human_request_created_at"
    HUMAN_REQUEST_EXPIRES_AT = "human_request_expires_at"

    ESCALATION_CHAIN = "escalation_chain"
    ESCALATION_LEVEL = "escalation_level"
    ESCALATION_TARGET = "escalation_target"

    REQUIRE_HUMAN_APPROVAL = "require_human_approval"
    REQUIRE_HUMAN_ON_CRITICAL = "require_human_on_critical"
    HIGH_RISK = "high_risk"

    LONG_RUNNING = "long_running"
    LONG_RUNNING_NOTIFY_CHANNEL = "long_running_notify_channel"
    LONG_RUNNING_CHECKPOINT_ON_PAUSE = "long_running_checkpoint_on_pause"
    RESUME_TOKEN = "resume_token"

    CHECKPOINT_ID = "checkpoint_id"
    PROGRESS_MESSAGE = "progress_message"

    SHARED_TASK_CONTEXT = "shared_task_context"


class TaskOrchestrationMetadataKey(StrEnum):
    """Classification and graph orchestration flat keys."""

    CLASSIFICATION = "classification"
    REQUESTED_CAPABILITY = "requested_capability"
    UNSUPPORTED_REASON = "unsupported_reason"
    RISK_LEVEL = "risk_level"
    CLASSIFICATION_CONFIDENCE = "classification_confidence"
    CLASSIFICATION_RATIONALE = "classification_rationale"
    CLASSIFIER_SOURCE = "classifier_source"
    PLAN_ID = "plan_id"
    GRAPH_ID = "graph_id"
    NEEDS_MORE_INFORMATION = "needs_more_information"


class TaskResultMetadataKey(StrEnum):
    """Task result summary flat keys (``TaskResult.metadata``)."""

    TASK_RESULT = "task_result.v1"
    VALIDATION_VALID = "validation_valid"
    VALIDATION_ERRORS = "validation_errors"
    VALIDATION_WARNINGS = "validation_warnings"
    EXECUTION_COST = "execution_cost"
    EXECUTION_TOTAL_TOKENS = "execution_total_tokens"
    RUNTIME_EVENTS = "runtime_events"
    TASK_TRACE_EVENTS = "task_trace_events"
    GRAPH_NODE_COUNT = "graph_node_count"
    AGENT_COUNT = "agent_count"
    AGENT_IDS = "agent_ids"
    RETRY_COUNT = "retry_count"
    RETRIES = "retries"
    ALL_COMPLETED = "all_completed"
    SHADOW_WORKSPACE_ID = "shadow_workspace_id"
    SHADOW_ARTIFACT_COUNT = "shadow_artifact_count"
    SANDBOX_SESSION_ID = "sandbox_session_id"
    SANDBOX_OPERATION_COUNT = "sandbox_operation_count"
    APPLICATION_RUN_SUMMARY = "application_run_summary.v1"


TASK_METADATA_LEGACY_OPTION_KEYS: frozenset[str] = frozenset(
    key.value
    for key in (
        TaskMetadataKey.HUMAN_RESPONSE,
        TaskMetadataKey.HUMAN_DECISION,
        TaskMetadataKey.HUMAN_APPROVED,
        TaskMetadataKey.HUMAN_REJECTED,
        TaskMetadataKey.HUMAN_ESCALATED,
        TaskMetadataKey.REQUIRE_HUMAN_APPROVAL,
        TaskMetadataKey.REQUIRE_HUMAN_ON_CRITICAL,
        TaskMetadataKey.HIGH_RISK,
        TaskMetadataKey.LONG_RUNNING,
        TaskMetadataKey.LONG_RUNNING_NOTIFY_CHANNEL,
        TaskMetadataKey.LONG_RUNNING_CHECKPOINT_ON_PAUSE,
        TaskMetadataKey.RESUME_TOKEN,
    )
)

TASK_METADATA_LEGACY_RUNTIME_KEYS: frozenset[str] = frozenset(
    key.value
    for key in (
        *TaskOrchestrationMetadataKey,
        TaskMetadataKey.GOVERNANCE_PAUSE,
        TaskMetadataKey.GOVERNANCE_HUMAN_REQUEST,
        TaskMetadataKey.GOVERNANCE_INTERRUPT,
        TaskMetadataKey.GOVERNANCE_PAUSE_RECORD,
        TaskMetadataKey.HUMAN_REQUEST_CREATED_AT,
        TaskMetadataKey.HUMAN_REQUEST_EXPIRES_AT,
        TaskMetadataKey.ESCALATION_LEVEL,
        TaskMetadataKey.ESCALATION_TARGET,
        TaskMetadataKey.ESCALATION_CHAIN,
        TaskMetadataKey.CHECKPOINT_ID,
        TaskMetadataKey.PROGRESS_MESSAGE,
    )
)

# Backward-compatible aliases (prefer ``TaskMetadataKey`` members directly).
GOVERNANCE_HUMAN_REQUEST_KEY = TaskMetadataKey.GOVERNANCE_HUMAN_REQUEST
GOVERNANCE_INTERRUPT_KEY = TaskMetadataKey.GOVERNANCE_INTERRUPT
GOVERNANCE_PAUSE_KEY = TaskMetadataKey.GOVERNANCE_PAUSE
GOVERNANCE_PAUSE_RECORD_KEY = TaskMetadataKey.GOVERNANCE_PAUSE_RECORD
HUMAN_APPROVED_KEY = TaskMetadataKey.HUMAN_APPROVED
HUMAN_DECISION_KEY = TaskMetadataKey.HUMAN_DECISION
HUMAN_ESCALATED_KEY = TaskMetadataKey.HUMAN_ESCALATED
HUMAN_REJECTED_KEY = TaskMetadataKey.HUMAN_REJECTED
HUMAN_RESPONSE_KEY = TaskMetadataKey.HUMAN_RESPONSE
HUMAN_REQUEST_CREATED_AT_KEY = TaskMetadataKey.HUMAN_REQUEST_CREATED_AT
HUMAN_REQUEST_EXPIRES_AT_KEY = TaskMetadataKey.HUMAN_REQUEST_EXPIRES_AT
ESCALATION_CHAIN_KEY = TaskMetadataKey.ESCALATION_CHAIN
ESCALATION_LEVEL_KEY = TaskMetadataKey.ESCALATION_LEVEL
ESCALATION_TARGET_KEY = TaskMetadataKey.ESCALATION_TARGET
LONG_RUNNING_FLAG = TaskMetadataKey.LONG_RUNNING
LONG_RUNNING_NOTIFY_CHANNEL_KEY = TaskMetadataKey.LONG_RUNNING_NOTIFY_CHANNEL
LONG_RUNNING_CHECKPOINT_ON_PAUSE_KEY = TaskMetadataKey.LONG_RUNNING_CHECKPOINT_ON_PAUSE
RESUME_TOKEN_KEY = TaskMetadataKey.RESUME_TOKEN
CHECKPOINT_ID_KEY = TaskMetadataKey.CHECKPOINT_ID
PROGRESS_MESSAGE_KEY = TaskMetadataKey.PROGRESS_MESSAGE
SHARED_TASK_CONTEXT_KEY = TaskMetadataKey.SHARED_TASK_CONTEXT
