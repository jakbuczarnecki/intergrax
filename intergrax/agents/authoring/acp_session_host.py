# © Artur Czarnecki. All rights reserved.

"""Host context passed into direct ACP runs (Tier-3 slices)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from intergrax.agents.authoring.acp_runtime_session_ports import AcpRuntimeSessionHooks
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.agent_run_binding import AgentRunBinding
from intergrax.contracts.budget_reaction_hook import (
    BudgetReactionHook,
    CustomBudgetReactionHook,
)
from intergrax.contracts.execution_bound_declarative_tool_invocation import (
    ExecutionBoundDeclarativeToolInvoker,
)
from intergrax.contracts.runtime_environment import RuntimeEnvironmentProfile
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.runtime.decision_flow import DecisionFlowGate
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedgerFactory
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter

ACP_HOST_CONTEXT_KEY = AcpMetadataKey.HOST_CONTEXT


class ACPSessionHostContext(BaseModel):
    """Optional Tier-3 host slices for merge_environment on direct run."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    runtime_profile: RuntimeEnvironmentProfile | None = None
    binding: AgentRunBinding | None = Field(default=None, exclude=True)
    declarative_tool_invoker: SkipValidation[
        ExecutionBoundDeclarativeToolInvoker | None
    ] = Field(default=None, exclude=True)
    decision_flow_gate: SkipValidation[DecisionFlowGate[AgentExecutionResult] | None] = (
        Field(default=None, exclude=True)
    )
    notification_adapter: SkipValidation[NotificationAdapter | None] = Field(
        default=None,
        exclude=True,
    )
    budget_reaction_hook: SkipValidation[
        BudgetReactionHook | CustomBudgetReactionHook | None
    ] = Field(default=None, exclude=True)
    execution_budget_ledger_factory: SkipValidation[ExecutionBudgetLedgerFactory | None] = Field(
        default=None,
        exclude=True,
    )
    runtime_session_hooks: SkipValidation[AcpRuntimeSessionHooks | None] = Field(
        default=None,
        exclude=True,
    )
