# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Strongly typed task intake, runtime state and result contracts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.execution_interrupt import ExecutionInterrupt

TASK_CONTRACT_METADATA_KEY = "task_contract.v1"
VERDICT_APPROVE = "approve"
VERDICT_REJECT = "reject"
VERDICT_ESCALATE = "escalate"


class TaskIsolationOptions(BaseModel):
    shadow_workspace: bool = False
    shadow_workspace_cleanup: bool = False
    sandbox: bool = False
    sandbox_cleanup: bool = False


class TaskHumanInput(BaseModel):
    response_text: Optional[str] = None
    verdict: Optional[str] = None

    @property
    def is_resumed(self) -> bool:
        return self.verdict == VERDICT_APPROVE

    @property
    def is_rejected(self) -> bool:
        return self.verdict == VERDICT_REJECT

    @property
    def is_escalated(self) -> bool:
        return self.verdict == VERDICT_ESCALATE


class TaskGovernanceOptions(BaseModel):
    require_human_approval: bool = False
    require_human_on_critical: bool = True
    high_risk: bool = False


class TaskExecutionOptions(BaseModel):
    """User-provided intake options for a task."""

    isolation: TaskIsolationOptions = Field(default_factory=TaskIsolationOptions)
    human: TaskHumanInput = Field(default_factory=TaskHumanInput)
    governance: TaskGovernanceOptions = Field(default_factory=TaskGovernanceOptions)


class EscalationStep(BaseModel):
    level: int
    target: str
    message: str = ""


class TaskPauseRecord(BaseModel):
    pause_id: str
    task_id: str
    human_request_id: str
    reason: str = ""
    created_at: str = ""
    schema_version: str = "pause_record.v1"


class TaskGovernanceState(BaseModel):
    paused: bool = False
    human_request: Optional[HumanRequest] = None
    execution_interrupt: Optional[ExecutionInterrupt] = None
    pause_record: Optional[TaskPauseRecord] = None
    escalation_level: int = 0
    escalation_target: Optional[str] = None
    escalation_chain: List[EscalationStep] = Field(default_factory=list)


class TaskClassificationState(BaseModel):
    value: Optional[str] = None
    requested_capability: Optional[str] = None
    unsupported_reason: Optional[str] = None
    risk_level: Optional[str] = None


class TaskOrchestrationState(BaseModel):
    plan_id: Optional[str] = None
    graph_id: Optional[str] = None
    needs_more_information: bool = False


class TaskRuntimeState(BaseModel):
    classification: TaskClassificationState = Field(default_factory=TaskClassificationState)
    orchestration: TaskOrchestrationState = Field(default_factory=TaskOrchestrationState)
    governance: TaskGovernanceState = Field(default_factory=TaskGovernanceState)


class TaskValidationSummary(BaseModel):
    valid: bool = False
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class TaskExecutionMetrics(BaseModel):
    cost: float = 0.0
    total_tokens: int = 0
    runtime_events: int = 0
    task_trace_events: int = 0


class TaskIsolationSummary(BaseModel):
    shadow_workspace_id: Optional[str] = None
    shadow_artifact_count: Optional[int] = None
    sandbox_session_id: Optional[str] = None
    sandbox_operation_count: Optional[int] = None


class TaskRetryRecord(BaseModel):
    attempt: int
    agent_id: str
    alternate_agent_id: Optional[str] = None
    reason: str = ""


class TaskOrchestrationSummary(BaseModel):
    classification: str = ""
    plan_id: str = ""
    graph_id: str = ""
    graph_node_count: int = 0
    agent_count: int = 0
    agent_ids: List[str] = Field(default_factory=list)
    retry_count: int = 0
    retries: List[TaskRetryRecord] = Field(default_factory=list)
    all_completed: bool = False


class TaskResultSummary(BaseModel):
    validation: TaskValidationSummary = Field(default_factory=TaskValidationSummary)
    metrics: TaskExecutionMetrics = Field(default_factory=TaskExecutionMetrics)
    isolation: TaskIsolationSummary = Field(default_factory=TaskIsolationSummary)
    orchestration: TaskOrchestrationSummary = Field(default_factory=TaskOrchestrationSummary)
    escalation_level: int = 0
    escalation_chain: List[EscalationStep] = Field(default_factory=list)
    governance_human_request: Optional[Dict[str, Any]] = None


class TaskContractPayload(BaseModel):
    """Serialized typed contract embedded in metadata for cross-layer transport."""

    options: TaskExecutionOptions = Field(default_factory=TaskExecutionOptions)
    runtime: TaskRuntimeState = Field(default_factory=TaskRuntimeState)
