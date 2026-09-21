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
from intergrax.contracts.execution_failure_evidence import ExecutionFailureKind
from intergrax.contracts.execution_identity import validate_execution_id
from intergrax.contracts.external_operations.failure import ExternalOperationFailureKind
from intergrax.runtime.events.payloads.canonical import (
    AgentSelectionPayloadV1,
    ContextAssemblyPayloadV1,
    ContextAssemblyPayloadV2,
    ContextCandidatePayloadV1,
    DecisionPayloadV1,
    DelegationGrantedPayloadV1,
    ExecutionFailurePayloadV1,
    ExternalOperationFailurePayloadV1,
    GraphNodePayloadV1,
    HandoffPayloadV1,
    HumanPayloadV1,
    InterruptPayloadV1,
    LlmCallPayloadV1,
    SkillResolvedPayloadV1,
    TaskLifecyclePayloadV1,
    ToolPayloadV1,
    ValidationPayloadV1,
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


def _tuple_str_field(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    return ()


def _validation_payload_from_raw(
    raw: dict[str, Any],
    *,
    default_valid: bool | None = None,
) -> ValidationPayloadV1:
    errors = raw.get("errors")
    rule_ids: tuple[str, ...] = ()
    if isinstance(errors, list):
        rule_ids = tuple(str(item) for item in errors if item is not None and str(item))
    else:
        rule_ids = _tuple_str_field(raw, "rule_ids_failed")
        if not rule_ids:
            error_type = raw.get("error_type")
            if error_type is not None and str(error_type):
                rule_ids = (str(error_type),)
    error_count = raw.get("error_count")
    if error_count is None:
        error_count = len(rule_ids) if rule_ids else 0
    valid = raw.get("valid")
    if valid is None:
        if default_valid is not None:
            valid = default_valid
        else:
            valid = int(error_count) == 0
    return ValidationPayloadV1(
        valid=bool(valid),
        error_count=int(error_count),
        warning_count=_int_field(raw, "warning_count"),
        stage=_str_field(raw, "stage", _str_field(raw, "step_name")),
        rule_ids_failed=rule_ids,
    )


def _context_assembly_v1_from_raw(raw: dict[str, Any]) -> ContextAssemblyPayloadV1:
    summary = raw.get("summary_tier")
    return ContextAssemblyPayloadV1(
        node_id=_str_field(raw, "node_id"),
        summary_tier=str(summary) if summary is not None else None,
        context_original_chars=_int_field(raw, "context_original_chars"),
        context_final_chars=_int_field(raw, "context_final_chars"),
        trimmed=bool(raw.get("trimmed", False)),
        engine_id=_str_field(raw, "engine_id"),
    )


def _context_assembly_v2_from_raw(raw: dict[str, Any]) -> ContextAssemblyPayloadV2:
    summary = raw.get("summary_tier")
    step_index = raw.get("step_index")
    step_kind = raw.get("step_kind")
    return ContextAssemblyPayloadV2(
        node_id=_str_field(raw, "node_id"),
        summary_tier=str(summary) if summary is not None else None,
        context_original_chars=_int_field(raw, "context_original_chars"),
        context_final_chars=_int_field(raw, "context_final_chars"),
        trimmed=bool(raw.get("trimmed", False)),
        engine_id=_str_field(raw, "engine_id"),
        step_index=int(step_index) if step_index is not None else None,
        step_kind=str(step_kind) if step_kind is not None else None,
        fragment_token_cost=_int_field(raw, "fragment_token_cost"),
        estimated_cost_microusd=_int_field(raw, "estimated_cost_microusd"),
    )


def _human_payload_from_raw(raw: dict[str, Any]) -> HumanPayloadV1:
    nested = raw.get("human_request")
    request_id = _str_field(raw, "request_id")
    if not request_id and isinstance(nested, dict):
        request_id = _str_field(nested, "request_id")
    if not request_id:
        request_id = _str_field(raw, "human_request_id")
    option_selected = _str_field(
        raw,
        "option_selected",
        _str_field(raw, "verdict", _str_field(raw, "decision")),
    )
    if not option_selected and isinstance(nested, dict):
        options = nested.get("options")
        if isinstance(options, list) and options:
            option_selected = str(options[0])
    if not request_id:
        raise ValueError("human approval payload missing request_id")
    if not option_selected:
        option_selected = "pending"
    operator_id = raw.get("operator_id") or raw.get("approver_user_id")
    comment = raw.get("comment") or raw.get("response")
    return HumanPayloadV1(
        request_id=request_id,
        option_selected=option_selected,
        operator_id=str(operator_id) if operator_id is not None else None,
        comment=str(comment) if comment is not None else None,
    )


def _decision_payload_from_raw(raw: dict[str, Any]) -> DecisionPayloadV1:
    decision_type = _str_field(raw, "decision_type", _str_field(raw, "decision"))
    record = raw.get("decision_record")
    if not decision_type and isinstance(record, dict):
        decision_type = _str_field(record, "decision_type")
    reason = _str_field(raw, "reason")
    if not reason and isinstance(record, dict):
        reason = _str_field(record, "rationale")
    if not decision_type:
        raise ValueError("decision_emitted payload missing decision_type")
    if not reason:
        raise ValueError("decision_emitted payload missing reason")
    severity = _str_field(raw, "severity", "info")
    interrupt_id = raw.get("interrupt_id")
    return DecisionPayloadV1(
        decision_type=decision_type,
        reason=reason,
        severity=severity,
        interrupt_id=str(interrupt_id) if interrupt_id is not None else None,
    )


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

    if event_type == RuntimeEventType.DECISION_EMITTED:
        typed = _decision_payload_from_raw(raw)
        promote = {k: raw[k] for k in ("step_id", "policy_action", "decision_record") if k in raw}
        return typed, promote or None

    if event_type == RuntimeEventType.AGENT_SELECTED:
        selected = _str_field(raw, "selected_agent_id")
        if not selected:
            raise ValueError("agent_selected payload missing selected_agent_id")
        typed = AgentSelectionPayloadV1(
            requested_agent_id=_str_field(raw, "requested_agent_id"),
            selected_agent_id=selected,
            capability=_str_field(raw, "capability"),
            match_score=float(raw["match_score"]) if raw.get("match_score") is not None else None,
            selection_reason=_str_field(raw, "selection_reason"),
            fallback_used=bool(raw.get("fallback_used", False)),
        )
        promote = {
            k: raw[k]
            for k in ("selected_agent_id", "selection_reason")
            if k in raw
        }
        return typed, promote or None

    if event_type == RuntimeEventType.CONTEXT_BUILT:
        typed = _context_assembly_v1_from_raw(raw)
        return typed, dict(raw) if raw else None

    if event_type in {RuntimeEventType.CONTEXT_ASSEMBLED, RuntimeEventType.CONTEXT_TRIMMED}:
        typed = _context_assembly_v2_from_raw(raw)
        return typed, dict(raw) if raw else None

    if event_type in {
        RuntimeEventType.CONTEXT_CANDIDATE_COLLECTED,
        RuntimeEventType.CONTEXT_CANDIDATE_DROPPED,
    }:
        typed = ContextCandidatePayloadV1(
            provider_id=_str_field(raw, "provider_id"),
            fragment_count=_int_field(raw, "fragment_count"),
            engine_id=_str_field(raw, "engine_id"),
            drop_reason=_str_field(raw, "drop_reason"),
            provider_version=_str_field(raw, "provider_version"),
        )
        if not typed.provider_id:
            raise ValueError(f"{event_type.value} payload missing provider_id")
        return typed, dict(raw) if raw else None

    if event_type == RuntimeEventType.CONTEXT_VALIDATION_FAILED:
        typed = _validation_payload_from_raw(raw, default_valid=False)
        return typed, dict(raw) if raw else None

    if event_type == RuntimeEventType.SKILL_RESOLVED:
        typed = SkillResolvedPayloadV1(
            skill_ids=_tuple_str_field(raw, "skill_ids"),
            tool_ids=_tuple_str_field(raw, "tool_ids"),
            prompt_instruction_ids=_tuple_str_field(raw, "prompt_instruction_ids"),
            policy_fragment_ids=_tuple_str_field(raw, "policy_fragment_ids"),
            risk_tier=_str_field(raw, "risk_tier", "unknown"),
        )
        return typed, dict(raw) if raw else None

    if event_type == RuntimeEventType.STEP_FAILED:
        if not raw:
            typed = ValidationPayloadV1(valid=False, error_count=1, stage="")
        else:
            typed = _validation_payload_from_raw(raw, default_valid=False)
            if typed.error_count == 0 and not typed.valid:
                typed = typed.model_copy(update={"error_count": 1})
        return typed, dict(raw) if raw else None

    if event_type == RuntimeEventType.EXECUTION_FAILED:
        kind_raw = _str_field(raw, "failure_kind")
        if not kind_raw:
            raise ValueError("execution_failed payload missing failure_kind")
        typed = ExecutionFailurePayloadV1(
            failure_kind=ExecutionFailureKind(kind_raw),
            safe_summary=_str_field(raw, "safe_summary", _str_field(raw, "message")),
            failure_code=raw.get("failure_code") if isinstance(raw.get("failure_code"), str) else None,
        )
        return typed, dict(raw) if raw else None

    if event_type == RuntimeEventType.EXTERNAL_OPERATION_FAILED:
        execution_id_raw = raw.get("execution_id")
        if execution_id_raw is None:
            raise ValueError("external_operation_failed payload missing execution_id")
        kind_raw = _str_field(raw, "failure_kind")
        if not kind_raw:
            raise ValueError("external_operation_failed payload missing failure_kind")
        typed = ExternalOperationFailurePayloadV1(
            execution_id=validate_execution_id(execution_id_raw),
            operation_attempt_id=_str_field(raw, "operation_attempt_id", "unknown"),
            provider_id=_str_field(raw, "provider_id", "unknown"),
            operation_type=_str_field(raw, "operation_type", "unknown"),
            failure_kind=ExternalOperationFailureKind(kind_raw),
            retryable=bool(raw.get("retryable", False)),
            evidence_refs=_tuple_str_field(raw, "evidence_refs"),
        )
        return typed, dict(raw) if raw else None

    if event_type == RuntimeEventType.VALIDATION_STARTED:
        typed = _validation_payload_from_raw(raw, default_valid=True)
        return typed, dict(raw) if raw else None

    if event_type == RuntimeEventType.VALIDATION_PASSED:
        typed = _validation_payload_from_raw(raw, default_valid=True)
        return typed, dict(raw) if raw else None

    if event_type == RuntimeEventType.VALIDATION_FAILED:
        typed = _validation_payload_from_raw(raw, default_valid=False)
        return typed, dict(raw) if raw else None

    if event_type in {RuntimeEventType.HUMAN_APPROVAL_REQUESTED, RuntimeEventType.HUMAN_APPROVAL_RECEIVED}:
        typed = _human_payload_from_raw(raw)
        return typed, dict(raw) if raw else None

    if event_type in {RuntimeEventType.HANDOFF_INITIATED, RuntimeEventType.HANDOFF_COMPLETED}:
        from_agent = _str_field(raw, "from_agent", _str_field(raw, "from_agent_id"))
        to_agent = _str_field(raw, "to_agent", _str_field(raw, "to_agent_id"))
        if not from_agent or not to_agent:
            raise ValueError(f"{event_type.value} payload missing handoff agent ids")
        typed = HandoffPayloadV1(
            from_agent=from_agent,
            to_agent=to_agent,
            capability=_str_field(raw, "capability", _str_field(raw, "to_capability")),
            artifact_ids=_tuple_str_field(raw, "artifact_ids"),
        )
        return typed, dict(raw) if raw else None

    if event_type == RuntimeEventType.DELEGATION_GRANTED:
        parent = _str_field(raw, "parent_agent_id")
        child = _str_field(raw, "child_agent_id")
        node_id = _str_field(raw, "node_id")
        if not parent or not child or not node_id:
            raise ValueError("delegation_granted payload missing parent_agent_id/child_agent_id/node_id")
        typed = DelegationGrantedPayloadV1(
            parent_agent_id=parent,
            child_agent_id=child,
            node_id=node_id,
            rationale=_str_field(raw, "rationale"),
            requested_permission_scopes=_tuple_str_field(raw, "requested_permission_scopes"),
            effective_permission_scopes=_tuple_str_field(raw, "effective_permission_scopes"),
            permission_scopes=_tuple_str_field(raw, "permission_scopes"),
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
