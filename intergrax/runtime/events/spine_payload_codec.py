# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Central legacy spine payload → typed envelope encoding (OBS-DIAG-EC1)."""

from __future__ import annotations

import json
from typing import Any

from intergrax.contracts.application_observability_attributes import (
    coerce_observability_attribute_mapping,
)
from intergrax.runtime.events.payload_registry import merge_payload_envelope, runtime_event_with_payload
from intergrax.runtime.events.payloads import RuntimeEventPayload
from intergrax.runtime.events.payloads.canonical import (
    GraphNodePayloadV1,
    InterruptPayloadV1,
    LlmCallPayloadV1,
    TaskLifecyclePayloadV1,
    ToolPayloadV1,
)
from intergrax.runtime.events.payloads.spine_families import (
    BudgetSignalPayloadV1,
    CancellationLifecyclePayloadV1,
    GraphBackpressurePayloadV1,
    GuardrailBlockedPayloadV1,
    HumanTimeoutPayloadV1,
    MemoryAccessPayloadV1,
    OperationalAlertPayloadV1,
    PauseLifecyclePayloadV1,
    PlanLifecyclePayloadV1,
    PolicyDecisionSpinePayloadV1,
    RetryLifecyclePayloadV1,
    TaskProgressPayloadV1,
    TracePersistedPayloadV1,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.runtime_event_payload_policy import (
    PayloadWriteMode,
    get_runtime_event_payload_policy,
)


def _aux_from_raw(raw: dict[str, Any], *, exclude: frozenset[str]) -> dict[str, Any]:
    leftover = {k: v for k, v in raw.items() if k not in exclude}
    return coerce_observability_attribute_mapping(leftover)


def _str_field(raw: dict[str, Any], key: str, default: str = "") -> str:
    value = raw.get(key, default)
    return str(value) if value is not None else default


def _int_field(raw: dict[str, Any], key: str, default: int = 0) -> int:
    value = raw.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def legacy_spine_payload_to_typed(
    event_type: RuntimeEventType,
    raw: dict[str, Any],
) -> tuple[RuntimeEventPayload, dict[str, Any] | None]:
    """Map unstructured spine payload dicts to registered payload models."""
    if event_type in {
        RuntimeEventType.TASK_CREATED,
        RuntimeEventType.TASK_CLASSIFIED,
        RuntimeEventType.TASK_COMPLETED,
        RuntimeEventType.TASK_FAILED,
    }:
        typed = TaskLifecyclePayloadV1(
            task_state=_str_field(raw, "task_state"),
            message=_str_field(raw, "message"),
            capability=_str_field(raw, "capability"),
            source=_str_field(raw, "source", "task_lifecycle"),
        )
        promote = {
            k: raw[k]
            for k in ("task_state", "message", "capability", "source")
            if k in raw
        }
        return typed, promote or None

    if event_type in {RuntimeEventType.PLAN_CREATED, RuntimeEventType.PLAN_UPDATED, RuntimeEventType.PLAN_FAILED}:
        exclude = frozenset(
            {
                "plan_id",
                "step_count",
                "task_state",
                "message",
                "failure_kind",
                "error_type",
                "error_message",
                "raw_hash",
                "decision_record",
            }
        )
        typed = PlanLifecyclePayloadV1(
            plan_id=_str_field(raw, "plan_id"),
            step_count=_int_field(raw, "step_count"),
            task_state=_str_field(raw, "task_state"),
            message=_str_field(raw, "message"),
            failure_kind=_str_field(raw, "failure_kind"),
            error_type=_str_field(raw, "error_type"),
            error_message=_str_field(raw, "error_message"),
            raw_hash=raw.get("raw_hash") if isinstance(raw.get("raw_hash"), str) else None,
            auxiliary=_aux_from_raw(raw, exclude=exclude),
        )
        promote = {k: raw[k] for k in ("plan_id", "step_count", "task_state") if k in raw}
        if "decision_record" in raw:
            promote["decision_record"] = raw["decision_record"]
        return typed, promote or None

    if event_type in {
        RuntimeEventType.PAUSE_REQUESTED,
        RuntimeEventType.PAUSED,
        RuntimeEventType.RESUMED,
    }:
        exclude = frozenset(
            {
                "lifecycle_state",
                "checkpoint_id",
                "resume_token",
                "progress_message",
                "reason",
            }
        )
        typed = PauseLifecyclePayloadV1(
            lifecycle_state=_str_field(raw, "lifecycle_state", event_type.value),
            checkpoint_id=_str_field(raw, "checkpoint_id"),
            resume_token=_str_field(raw, "resume_token"),
            progress_message=_str_field(raw, "progress_message"),
            reason=_str_field(raw, "reason"),
            auxiliary=_aux_from_raw(raw, exclude=exclude),
        )
        return typed, dict(raw) if raw else None

    if event_type == RuntimeEventType.RETRY_SCHEDULED:
        typed = RetryLifecyclePayloadV1(
            scope=_str_field(raw, "scope"),
            attempt=_int_field(raw, "attempt"),
            max_retries=_int_field(raw, "max_retries"),
            reason=_str_field(raw, "reason"),
            alternate_agent_id=raw.get("alternate_agent_id")
            if isinstance(raw.get("alternate_agent_id"), str)
            else None,
        )
        return typed, dict(raw)

    if event_type == RuntimeEventType.RETRY_STARTED:
        typed = RetryLifecyclePayloadV1(
            scope=_str_field(raw, "scope"),
            retry_ordinal=_int_field(raw, "retry_ordinal"),
            reason=_str_field(raw, "reason"),
        )
        return typed, dict(raw)

    if event_type in {RuntimeEventType.CANCELLATION_REQUESTED, RuntimeEventType.CANCELLED}:
        typed = CancellationLifecyclePayloadV1(
            reason=_str_field(raw, "reason"),
            initiated_by=_str_field(raw, "initiated_by"),
            scope=_str_field(raw, "scope"),
        )
        return typed, dict(raw) if raw else None

    if event_type in {RuntimeEventType.MEMORY_READ, RuntimeEventType.MEMORY_WRITE}:
        exclude = frozenset({"namespace", "key", "found", "record_id", "operation"})
        typed = MemoryAccessPayloadV1(
            namespace=_str_field(raw, "namespace"),
            key=_str_field(raw, "key"),
            found=bool(raw.get("found", False)),
            record_id=raw.get("record_id") if isinstance(raw.get("record_id"), str) else None,
            operation=_str_field(raw, "operation", _str_field(raw, "write_policy")),
            auxiliary=_aux_from_raw(raw, exclude=exclude),
        )
        return typed, dict(raw)

    if event_type == RuntimeEventType.INGESTION_FAILED:
        exclude = frozenset(
            {
                "attachment_id",
                "session_id",
                "user_id",
                "error_type",
                "error_message",
                "alert_kind",
                "source",
            }
        )
        typed = OperationalAlertPayloadV1(
            alert_kind="ingestion_failed",
            source=_str_field(raw, "session_id"),
            reason=_str_field(raw, "user_id"),
            error_type=_str_field(raw, "error_type"),
            error_message=_str_field(raw, "error_message"),
            attachment_id=_str_field(raw, "attachment_id"),
            auxiliary=_aux_from_raw(raw, exclude=exclude),
        )
        return typed, dict(raw)

    if event_type == RuntimeEventType.SKILL_IMPORT_FAILED:
        typed = OperationalAlertPayloadV1(
            alert_kind="skill_import_failed",
            source=_str_field(raw, "source"),
            reason=_str_field(raw, "reason"),
        )
        return typed, dict(raw)

    if event_type in {RuntimeEventType.INTERRUPT_ESCALATED, RuntimeEventType.RUNTIME_HANDLER_FAILED}:
        typed = OperationalAlertPayloadV1(
            alert_kind=event_type.value,
            source=_str_field(raw, "source"),
            reason=_str_field(raw, "reason"),
            error_type=_str_field(raw, "error_type"),
            error_message=_str_field(raw, "error_message"),
            auxiliary=_aux_from_raw(raw, exclude=frozenset()),
        )
        return typed, dict(raw) if raw else None

    if event_type == RuntimeEventType.HUMAN_APPROVAL_TIMEOUT:
        typed = HumanTimeoutPayloadV1(
            request_id=_str_field(raw, "request_id"),
            timeout_reason=_str_field(raw, "timeout_reason", _str_field(raw, "reason")),
            elapsed_ms=_int_field(raw, "elapsed_ms"),
        )
        return typed, dict(raw) if raw else None

    if event_type == RuntimeEventType.POLICY_DECISION:
        material = raw.get("decision_material_ref")
        material_json = ""
        if material is not None:
            try:
                material_json = json.dumps(material, default=str)[:4096]
            except (TypeError, ValueError):
                material_json = ""
        typed = PolicyDecisionSpinePayloadV1(
            governance_evidence_schema=_str_field(raw, "governance_evidence_schema"),
            evidence_id=_str_field(raw, "evidence_id"),
            evaluation_point=_str_field(raw, "evaluation_point"),
            action=_str_field(raw, "action"),
            resource_type=_str_field(raw, "resource_type"),
            resource_scope=_str_field(raw, "resource_scope"),
            decision=_str_field(raw, "decision"),
            reason=_str_field(raw, "reason"),
            reason_code=_str_field(raw, "reason_code"),
            policy_bundle_id=_str_field(raw, "policy_bundle_id"),
            policy_bundle_version=_str_field(raw, "policy_bundle_version"),
            policy_bundle_digest=_str_field(raw, "policy_bundle_digest"),
            policy_rule_id=_str_field(raw, "policy_rule_id"),
            request_digest=_str_field(raw, "request_digest"),
            idempotency_key=_str_field(raw, "idempotency_key"),
            workspace_id=_str_field(raw, "workspace_id"),
            principal_id=_str_field(raw, "principal_id"),
            decision_material_ref_json=material_json,
            human_review_evidence_ref=_str_field(raw, "human_review_evidence_ref"),
        )
        return typed, dict(raw)

    if event_type == RuntimeEventType.BUDGET_EXCEEDED:
        typed = BudgetSignalPayloadV1(
            scope=_str_field(raw, "scope"),
            tokens_total=_int_field(raw, "tokens_total"),
            tokens_limit=_int_field(raw, "tokens_limit"),
            limit_source=_str_field(raw, "limit_source"),
            reaction=_str_field(raw, "reaction"),
            signal_kind="exceeded",
        )
        return typed, dict(raw)

    if event_type == RuntimeEventType.BUDGET_THRESHOLD:
        typed = BudgetSignalPayloadV1(
            scope=_str_field(raw, "scope"),
            tokens_total=_int_field(raw, "tokens_total"),
            tokens_limit=_int_field(raw, "tokens_limit"),
            ratio=float(raw.get("ratio", 0.0) or 0.0),
            limit_source=_str_field(raw, "limit_source"),
            signal_kind="threshold",
        )
        return typed, dict(raw)

    if event_type == RuntimeEventType.GRAPH_BACKPRESSURE:
        typed = GraphBackpressurePayloadV1(
            max_inflight_nodes=_int_field(raw, "max_inflight_nodes"),
            node_id=_str_field(raw, "node_id"),
        )
        return typed, dict(raw) if raw else None

    if event_type == RuntimeEventType.GUARDRAIL_BLOCKED:
        typed = GuardrailBlockedPayloadV1(
            scan_kind=_str_field(raw, "scan_kind"),
            hook=_str_field(raw, "hook"),
            backend_slug=_str_field(raw, "backend_slug"),
            reason=_str_field(raw, "reason"),
        )
        return typed, dict(raw)

    if event_type == RuntimeEventType.TRACE_PERSISTED:
        typed = TracePersistedPayloadV1(
            trace_ref=_str_field(raw, "trace_ref"),
            store_kind=_str_field(raw, "store_kind"),
            byte_count=_int_field(raw, "byte_count"),
        )
        return typed, dict(raw) if raw else None

    if event_type == RuntimeEventType.TASK_PROGRESS:
        reasons_raw = raw.get("reasons")
        reasons: tuple[str, ...] = ()
        if isinstance(reasons_raw, list):
            reasons = tuple(str(item) for item in reasons_raw)
        exclude = frozenset(
            {
                "event_kind",
                "progress_kind",
                "message",
                "selected_pattern",
                "reasons",
            }
        )
        typed = TaskProgressPayloadV1(
            progress_kind=_str_field(raw, "progress_kind", _str_field(raw, "event_kind")),
            message=_str_field(raw, "message"),
            selected_pattern=_str_field(raw, "selected_pattern"),
            reasons=reasons,
            auxiliary=_aux_from_raw(raw, exclude=exclude),
        )
        return typed, dict(raw)

    if event_type == RuntimeEventType.TOOL_REQUESTED:
        typed = ToolPayloadV1(
            tool_name=_str_field(raw, "tool_name", _str_field(raw, "tool_id")),
            status=_str_field(raw, "status", "requested"),
            duration_ms=_int_field(raw, "duration_ms"),
            redacted_input_summary=_str_field(raw, "redacted_input_summary"),
            step_id=_str_field(raw, "step_id"),
        )
        promote = {"tool_name": typed.tool_name} if typed.tool_name else None
        return typed, promote

    if event_type in {
        RuntimeEventType.TOOL_COMPLETED,
        RuntimeEventType.TOOL_DENIED,
        RuntimeEventType.TOOL_FAILED,
    }:
        status_map = {
            RuntimeEventType.TOOL_COMPLETED: "completed",
            RuntimeEventType.TOOL_DENIED: "denied",
            RuntimeEventType.TOOL_FAILED: "failed",
        }
        typed = ToolPayloadV1(
            tool_name=_str_field(raw, "tool_name", _str_field(raw, "tool_id")),
            status=_str_field(raw, "status", status_map[event_type]),
            duration_ms=_int_field(raw, "duration_ms"),
            redacted_input_summary=_str_field(raw, "redacted_input_summary"),
            step_id=_str_field(raw, "step_id"),
        )
        promote = {"tool_name": typed.tool_name} if typed.tool_name else None
        return typed, promote

    if event_type in {RuntimeEventType.STEP_STARTED, RuntimeEventType.STEP_COMPLETED}:
        typed = GraphNodePayloadV1(
            node_id=_str_field(raw, "node_id"),
            status=_str_field(raw, "status"),
            agent_id=_str_field(raw, "agent_id"),
            message=_str_field(raw, "message"),
        )
        promote = {"node_id": typed.node_id} if typed.node_id else None
        return typed, promote

    if event_type == RuntimeEventType.LLM_CALL:
        typed = LlmCallPayloadV1(
            model=_str_field(raw, "model", _str_field(raw, "model_id")),
            prompt_tokens=_int_field(raw, "prompt_tokens"),
            completion_tokens=_int_field(raw, "completion_tokens"),
            total_tokens=_int_field(raw, "total_tokens"),
            finish_reason=raw.get("finish_reason")
            if raw.get("finish_reason") is None or isinstance(raw.get("finish_reason"), str)
            else None,
            label=_str_field(raw, "label"),
        )
        return typed, dict(raw)

    if event_type == RuntimeEventType.INTERRUPT_REQUESTED or event_type == RuntimeEventType.INTERRUPT_HANDLED:
        metadata_raw = raw.get("metadata")
        metadata = coerce_observability_attribute_mapping(
            metadata_raw if isinstance(metadata_raw, dict) else {}
        )
        typed = InterruptPayloadV1(
            interrupt_type=_str_field(raw, "interrupt_type"),
            blocking=bool(raw.get("blocking", False)),
            recommended_action=_str_field(raw, "recommended_action"),
            metadata=metadata,
        )
        return typed, dict(raw) if raw else None

    policy = get_runtime_event_payload_policy(event_type)
    if policy.schema_id is None:
        raise ValueError(f"no legacy codec for event_type={event_type.value}")

    raise ValueError(f"legacy spine payload codec not implemented for {event_type.value}")


def prepare_canonical_production_write_event(event: RuntimeEvent) -> RuntimeEvent:
    """
    Upgrade legacy unstructured spine payloads to typed envelopes on canonical write.

    Historical read paths must not call this helper.
    """
    if event.payload.get("payload_schema_id") is not None:
        return event
    policy = get_runtime_event_payload_policy(event.event_type)
    if policy.write_mode == PayloadWriteMode.EXTENSION_EVENT_KIND:
        return event
    if policy.schema_id is None:
        return event
    typed, promote = legacy_spine_payload_to_typed(event.event_type, dict(event.payload))
    return runtime_event_with_payload(event, typed, promote_fields=promote)


__all__ = [
    "legacy_spine_payload_to_typed",
    "merge_payload_envelope",
    "prepare_canonical_production_write_event",
]
