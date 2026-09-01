"""Scenario application execution entry."""

from __future__ import annotations

import json
from dataclasses import dataclass

from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioExecutionRequest,
    execute_scenario_task,
)
from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace

from platform_proofs.scenarios.indirect_prompt_injection.application.execution_payload import (
    domain_payload_from_execution,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_models import (
    OrderProviderNote,
    OrderProviderOrder,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.order_workflow import (
    tool_trace_from_dict,
    tool_trace_to_dict,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.runtime_composition import (
    ORDER_ASSISTANT_CAPABILITY,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.run_bundle import (
    OrderAssistantRunBundle,
)

OUTCOME_RESOLVED = "RESOLVED"
OUTCOME_UNRESOLVED = "UNRESOLVED"


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


def _leak_scan_blob(*parts: object) -> str:
    return json.dumps(parts)


def _as_note_tuple(value: object) -> tuple[OrderProviderNote, ...]:
    if not isinstance(value, list):
        return ()
    notes: list[OrderProviderNote] = []
    for item in value:
        if isinstance(item, OrderProviderNote):
            notes.append(item)
        elif isinstance(item, dict):
            notes.append(OrderProviderNote.model_validate(item))
    return tuple(notes)


def _as_dict_list(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _as_str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _as_tool_traces(value: object) -> tuple[ToolCallTrace, ...]:
    if not isinstance(value, list):
        return ()
    traces: list[ToolCallTrace] = []
    for item in value:
        if isinstance(item, ToolCallTrace):
            traces.append(item)
        elif isinstance(item, dict):
            traces.append(tool_trace_from_dict(item))
    return tuple(traces)


def _order_state_from_domain(value: object) -> OrderProviderOrder | None:
    if isinstance(value, OrderProviderOrder):
        return value
    if isinstance(value, dict):
        return OrderProviderOrder.model_validate(value)
    return None


async def execute_order_assistant_run(bundle: OrderAssistantRunBundle) -> ScenarioExecutionResult:
    platform = bundle.runtime_composition.platform
    tenant_id = platform.tenant_id
    initial_order_state = bundle.provider_client.get_order(bundle.order_id)
    platform_result = await execute_scenario_task(
        platform,
        ScenarioExecutionRequest(
            tenant_id=tenant_id,
            message=bundle.user_message,
            capability=ORDER_ASSISTANT_CAPABILITY,
        ),
    )
    task_result = platform_result.task_result
    final_execution = task_result.execution_result
    if final_execution is None:
        raise RuntimeError("order_assistant_execution_missing")

    domain_payload = domain_payload_from_execution(final_execution)
    provider_state = bundle.provider_client.mutation_state()
    final_order_state = bundle.provider_client.get_order(bundle.order_id)

    return ScenarioExecutionResult(
        outcome=str(domain_payload.get("outcome", OUTCOME_UNRESOLVED)),
        terminal_summary=str(domain_payload.get("terminal_summary", "")),
        order_facts=dict(domain_payload.get("order_facts", {})),
        initial_order_state=initial_order_state,
        final_order_state=final_order_state,
        retrieved_notes=_as_note_tuple(domain_payload.get("retrieved_notes")),
        tool_traces=_as_tool_traces(domain_payload.get("tool_traces")),
        policy_evaluations=_as_dict_list(domain_payload.get("policy_evaluations")),
        planner_rounds=_as_dict_list(domain_payload.get("planner_rounds")),
        write_tool_proposed=bool(domain_payload.get("write_tool_proposed", False)),
        write_tool_executed=bool(domain_payload.get("write_tool_executed", False)),
        policy_denied=bool(domain_payload.get("policy_denied", False)),
        matched_policy_rule_ids=_as_str_tuple(domain_payload.get("matched_policy_rule_ids")),
        provider_write_count=provider_state.write_count,
        model_provider=str(domain_payload.get("model_provider", "unknown")),
        model_name=str(domain_payload.get("model_name", "unknown")),
        workflow_kind=str(domain_payload.get("workflow_kind", bundle.workflow.value)),
        leak_scan_blob=_leak_scan_blob(domain_payload),
        run_id=str(platform_result.run_id),
        tenant_id=tenant_id,
    )
