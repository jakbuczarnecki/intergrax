# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Additional canonical spine payload families (OBS-DIAG-EC1)."""

from __future__ import annotations

from pydantic import Field, field_validator

from intergrax.contracts.application_observability_attributes import ObservabilityAttributeValue
from intergrax.runtime.events.payloads.base import RuntimeEventPayload


def _coerce_auxiliary(value: object) -> dict[str, ObservabilityAttributeValue]:
    if not isinstance(value, dict):
        return {}
    from intergrax.contracts.application_observability_attributes import (
        coerce_observability_attribute_mapping,
    )

    return coerce_observability_attribute_mapping(value)


class PlanLifecyclePayloadV1(RuntimeEventPayload):
    schema_id = "plan_lifecycle.v1"

    plan_id: str = ""
    step_count: int = 0
    task_state: str = ""
    message: str = ""
    failure_kind: str = ""
    error_type: str = ""
    error_message: str = Field(default="", max_length=512)
    raw_hash: str | None = None
    auxiliary: dict[str, ObservabilityAttributeValue] = Field(default_factory=dict)

    @field_validator("auxiliary", mode="before")
    @classmethod
    def _validate_auxiliary(cls, value: object) -> dict[str, ObservabilityAttributeValue]:
        return _coerce_auxiliary(value)


class PauseLifecyclePayloadV1(RuntimeEventPayload):
    schema_id = "pause_lifecycle.v1"

    lifecycle_state: str = ""
    checkpoint_id: str = ""
    resume_token: str = ""
    progress_message: str = ""
    reason: str = ""
    auxiliary: dict[str, ObservabilityAttributeValue] = Field(default_factory=dict)

    @field_validator("auxiliary", mode="before")
    @classmethod
    def _validate_auxiliary(cls, value: object) -> dict[str, ObservabilityAttributeValue]:
        return _coerce_auxiliary(value)


class RetryLifecyclePayloadV1(RuntimeEventPayload):
    schema_id = "retry_lifecycle.v1"

    scope: str = ""
    attempt: int = 0
    max_retries: int = 0
    retry_ordinal: int = 0
    reason: str = ""
    alternate_agent_id: str | None = None


class CancellationLifecyclePayloadV1(RuntimeEventPayload):
    schema_id = "cancellation_lifecycle.v1"

    reason: str = ""
    initiated_by: str = ""
    scope: str = ""


class MemoryAccessPayloadV1(RuntimeEventPayload):
    schema_id = "memory_access.v1"

    namespace: str
    key: str
    found: bool
    record_id: str | None = None
    operation: str = ""
    auxiliary: dict[str, ObservabilityAttributeValue] = Field(default_factory=dict)

    @field_validator("auxiliary", mode="before")
    @classmethod
    def _validate_auxiliary(cls, value: object) -> dict[str, ObservabilityAttributeValue]:
        return _coerce_auxiliary(value)


class OperationalAlertPayloadV1(RuntimeEventPayload):
    schema_id = "operational_alert.v1"

    alert_kind: str = ""
    source: str = ""
    reason: str = ""
    error_type: str = ""
    error_message: str = Field(default="", max_length=512)
    attachment_id: str = ""
    session_id: str = ""
    auxiliary: dict[str, ObservabilityAttributeValue] = Field(default_factory=dict)

    @field_validator("auxiliary", mode="before")
    @classmethod
    def _validate_auxiliary(cls, value: object) -> dict[str, ObservabilityAttributeValue]:
        return _coerce_auxiliary(value)


class HumanTimeoutPayloadV1(RuntimeEventPayload):
    schema_id = "human_timeout.v1"

    request_id: str = ""
    timeout_reason: str = ""
    elapsed_ms: int = 0


class PolicyDecisionSpinePayloadV1(RuntimeEventPayload):
    schema_id = "policy_decision_spine.v1"

    governance_evidence_schema: str = ""
    evidence_id: str = ""
    evaluation_point: str = ""
    action: str = ""
    resource_type: str = ""
    resource_scope: str = ""
    decision: str = ""
    reason: str = ""
    reason_code: str = ""
    policy_bundle_id: str = ""
    policy_bundle_version: str = ""
    policy_bundle_digest: str = ""
    policy_rule_id: str = ""
    request_digest: str = ""
    idempotency_key: str = ""
    workspace_id: str = ""
    principal_id: str = ""
    decision_material_ref_json: str = Field(default="", max_length=4096)
    human_review_evidence_ref: str = ""


class BudgetSignalPayloadV1(RuntimeEventPayload):
    schema_id = "budget_signal.v1"

    scope: str = ""
    tokens_total: int = 0
    tokens_limit: int = 0
    ratio: float = 0.0
    limit_source: str = ""
    reaction: str = ""
    signal_kind: str = ""


class GraphBackpressurePayloadV1(RuntimeEventPayload):
    schema_id = "graph_backpressure.v1"

    max_inflight_nodes: int = 0
    node_id: str = ""


class GuardrailBlockedPayloadV1(RuntimeEventPayload):
    schema_id = "guardrail_blocked.v1"

    scan_kind: str = ""
    hook: str = ""
    backend_slug: str = ""
    reason: str = ""


class TracePersistedPayloadV1(RuntimeEventPayload):
    schema_id = "trace_persisted.v1"

    trace_ref: str = ""
    store_kind: str = ""
    byte_count: int = 0


class TaskProgressPayloadV1(RuntimeEventPayload):
    schema_id = "task_progress.v1"

    progress_kind: str = ""
    message: str = ""
    selected_pattern: str = ""
    reasons: tuple[str, ...] = ()
    auxiliary: dict[str, ObservabilityAttributeValue] = Field(default_factory=dict)

    @field_validator("auxiliary", mode="before")
    @classmethod
    def _validate_auxiliary(cls, value: object) -> dict[str, ObservabilityAttributeValue]:
        return _coerce_auxiliary(value)
