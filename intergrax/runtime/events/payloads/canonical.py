# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canon §42.23.1 payload families + OBS-BUS extension schemas."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from intergrax.runtime.events.payloads.base import RuntimeEventPayload


class DecisionPayloadV1(RuntimeEventPayload):
    schema_id = "decision.v1"

    decision_type: str
    reason: str
    severity: str = "info"
    interrupt_id: str | None = None


class ToolPayloadV1(RuntimeEventPayload):
    schema_id = "tool.v1"

    tool_name: str
    status: str
    duration_ms: int = 0
    redacted_input_summary: str = ""
    step_id: str = ""


class ValidationPayloadV1(RuntimeEventPayload):
    schema_id = "validation.v1"

    valid: bool
    error_count: int = 0
    warning_count: int = 0
    stage: str = ""
    rule_ids_failed: tuple[str, ...] = ()


class InterruptPayloadV1(RuntimeEventPayload):
    schema_id = "interrupt.v1"

    interrupt_type: str
    blocking: bool
    recommended_action: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanPayloadV1(RuntimeEventPayload):
    schema_id = "human.v1"

    request_id: str
    option_selected: str
    operator_id: str | None = None
    comment: str | None = None


class HandoffPayloadV1(RuntimeEventPayload):
    schema_id = "handoff.v1"

    from_agent: str
    to_agent: str
    capability: str = ""
    artifact_ids: tuple[str, ...] = ()


class DelegationGrantedPayloadV1(RuntimeEventPayload):
    schema_id = "delegation_granted.v1"

    parent_agent_id: str
    child_agent_id: str
    node_id: str
    rationale: str = ""
    permission_scopes: tuple[str, ...] = ()


class AgentSelectionPayloadV1(RuntimeEventPayload):
    schema_id = "agent_selection.v1"

    requested_agent_id: str = ""
    selected_agent_id: str
    capability: str = ""
    match_score: float | None = None
    selection_reason: str = ""
    fallback_used: bool = False


class GraphNodePayloadV1(RuntimeEventPayload):
    schema_id = "graph_node.v1"

    node_id: str
    status: str
    agent_id: str = ""
    message: str = ""


class LlmCallPayloadV1(RuntimeEventPayload):
    schema_id = "llm_call.v1"

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str | None = None
    label: str = ""


class TraceBridgePayloadV1(RuntimeEventPayload):
    """Wrapper for trace-bridged diagnostic detail."""

    schema_id = "trace_bridge.v1"

    trace_event_id: str
    trace_step: str
    trace_component: str
    trace_seq: int
    message: str
    source: str = "trace_bridge"
    diagnostic_schema_id: str = ""
    diagnostic_data: dict[str, Any] = Field(default_factory=dict)


class SkillResolvedPayloadV1(RuntimeEventPayload):
    schema_id = "skill_resolved.v1"

    skill_ids: tuple[str, ...]
    tool_ids: tuple[str, ...]
    prompt_instruction_ids: tuple[str, ...]
    policy_fragment_ids: tuple[str, ...]
    risk_tier: str


class ContextAssemblyPayloadV1(RuntimeEventPayload):
    schema_id = "context_assembly.v1"

    node_id: str
    summary_tier: str | None = None
    context_original_chars: int
    context_final_chars: int
    trimmed: bool = False


class TaskLifecyclePayloadV1(RuntimeEventPayload):
    schema_id = "task_lifecycle.v1"

    task_state: str
    message: str = ""
    capability: str = ""
    source: str = "task_lifecycle"
