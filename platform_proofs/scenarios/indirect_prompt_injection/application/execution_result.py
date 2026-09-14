"""Business execution result for the order assistant application layer."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace

from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_models import (
    OrderProviderNote,
)


@dataclass(frozen=True, slots=True)
class OrderAssistantExecutionResult:
    outcome: str
    terminal_summary: str
    order_facts: dict[str, object]
    retrieved_notes: tuple[OrderProviderNote, ...]
    tool_traces: tuple[ToolCallTrace, ...]
    policy_evaluations: tuple[dict[str, object], ...]
    planner_rounds: tuple[dict[str, object], ...]
    write_tool_proposed: bool
    write_tool_executed: bool
    policy_denied: bool
    matched_policy_rule_ids: tuple[str, ...]
    model_provider: str
    model_name: str
    workflow_kind: str
    leak_scan_blob: str
    run_id: str
    tenant_id: str
