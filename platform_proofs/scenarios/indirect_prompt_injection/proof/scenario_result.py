"""Proof-owned execution observation — application result plus provider measurements."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace

from platform_proofs.scenarios.indirect_prompt_injection.application.execution_result import (
    OrderAssistantExecutionResult,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_models import (
    OrderProviderNote,
    OrderProviderOrder,
)


@dataclass(frozen=True, slots=True)
class ScenarioExecutionResult:
    outcome: str
    terminal_summary: str
    order_facts: dict[str, object]
    initial_order_state: OrderProviderOrder | None
    final_order_state: OrderProviderOrder | None
    retrieved_notes: tuple[OrderProviderNote, ...]
    tool_traces: tuple[ToolCallTrace, ...]
    policy_evaluations: tuple[dict[str, object], ...]
    planner_rounds: tuple[dict[str, object], ...]
    write_tool_proposed: bool
    write_tool_executed: bool
    policy_denied: bool
    matched_policy_rule_ids: tuple[str, ...]
    provider_write_count: int
    model_provider: str
    model_name: str
    workflow_kind: str
    leak_scan_blob: str
    run_id: str
    tenant_id: str


def scenario_result_from_application(
    application: OrderAssistantExecutionResult,
    *,
    initial_order_state: OrderProviderOrder | None,
    final_order_state: OrderProviderOrder | None,
    provider_write_count: int,
) -> ScenarioExecutionResult:
    return ScenarioExecutionResult(
        outcome=application.outcome,
        terminal_summary=application.terminal_summary,
        order_facts=dict(application.order_facts),
        initial_order_state=initial_order_state,
        final_order_state=final_order_state,
        retrieved_notes=application.retrieved_notes,
        tool_traces=application.tool_traces,
        policy_evaluations=application.policy_evaluations,
        planner_rounds=application.planner_rounds,
        write_tool_proposed=application.write_tool_proposed,
        write_tool_executed=application.write_tool_executed,
        policy_denied=application.policy_denied,
        matched_policy_rule_ids=application.matched_policy_rule_ids,
        provider_write_count=provider_write_count,
        model_provider=application.model_provider,
        model_name=application.model_name,
        workflow_kind=application.workflow_kind,
        leak_scan_blob=application.leak_scan_blob,
        run_id=application.run_id,
        tenant_id=application.tenant_id,
    )
