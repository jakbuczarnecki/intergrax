# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime event payload schema registry (OBS-BUS-1)."""

from __future__ import annotations

from typing import Any, TypeVar

from intergrax.runtime.events.payloads import CANONICAL_PAYLOAD_TYPES, RuntimeEventPayload
from intergrax.runtime.events.payloads.canonical import (
    AgentSelectionPayloadV1,
    ContextAssemblyPayloadV1,
    DecisionPayloadV1,
    GraphNodePayloadV1,
    DelegationGrantedPayloadV1,
    HandoffPayloadV1,
    HumanPayloadV1,
    InterruptPayloadV1,
    LlmCallPayloadV1,
    SkillResolvedPayloadV1,
    TaskLifecyclePayloadV1,
    ToolPayloadV1,
    ValidationPayloadV1,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType

T = TypeVar("T", bound=RuntimeEventPayload)

_PAYLOAD_REGISTRY: dict[str, type[RuntimeEventPayload]] = {}
_EXTENSION_REGISTRY: dict[str, type[RuntimeEventPayload]] = {}


class RuntimeEventPayloadError(ValueError):
    """Raised when a runtime event payload envelope is invalid."""


class UnknownPayloadSchemaError(RuntimeEventPayloadError):
    """Raised when payload_schema_id is not registered."""


EVENT_TYPE_PREFERRED_SCHEMA: dict[RuntimeEventType, str] = {
    RuntimeEventType.AGENT_SELECTED: AgentSelectionPayloadV1.schema_id,
    RuntimeEventType.CONTEXT_ASSEMBLED: ContextAssemblyPayloadV1.schema_id,
    RuntimeEventType.CONTEXT_TRIMMED: ContextAssemblyPayloadV1.schema_id,
    RuntimeEventType.DECISION_EMITTED: DecisionPayloadV1.schema_id,
    RuntimeEventType.DELEGATION_GRANTED: DelegationGrantedPayloadV1.schema_id,
    RuntimeEventType.HANDOFF_COMPLETED: HandoffPayloadV1.schema_id,
    RuntimeEventType.HANDOFF_INITIATED: HandoffPayloadV1.schema_id,
    RuntimeEventType.HUMAN_APPROVAL_RECEIVED: HumanPayloadV1.schema_id,
    RuntimeEventType.HUMAN_APPROVAL_REQUESTED: HumanPayloadV1.schema_id,
    RuntimeEventType.INTERRUPT_HANDLED: InterruptPayloadV1.schema_id,
    RuntimeEventType.INTERRUPT_REQUESTED: InterruptPayloadV1.schema_id,
    RuntimeEventType.LLM_CALL: LlmCallPayloadV1.schema_id,
    RuntimeEventType.SKILL_RESOLVED: SkillResolvedPayloadV1.schema_id,
    RuntimeEventType.STEP_COMPLETED: GraphNodePayloadV1.schema_id,
    RuntimeEventType.STEP_FAILED: ValidationPayloadV1.schema_id,
    RuntimeEventType.STEP_STARTED: GraphNodePayloadV1.schema_id,
    RuntimeEventType.TASK_CLASSIFIED: TaskLifecyclePayloadV1.schema_id,
    RuntimeEventType.TASK_COMPLETED: TaskLifecyclePayloadV1.schema_id,
    RuntimeEventType.TASK_CREATED: TaskLifecyclePayloadV1.schema_id,
    RuntimeEventType.TASK_FAILED: TaskLifecyclePayloadV1.schema_id,
    RuntimeEventType.TOOL_COMPLETED: ToolPayloadV1.schema_id,
    RuntimeEventType.TOOL_DENIED: ToolPayloadV1.schema_id,
    RuntimeEventType.TOOL_FAILED: ToolPayloadV1.schema_id,
    RuntimeEventType.TOOL_REQUESTED: ToolPayloadV1.schema_id,
    RuntimeEventType.VALIDATION_FAILED: ValidationPayloadV1.schema_id,
    RuntimeEventType.VALIDATION_PASSED: ValidationPayloadV1.schema_id,
    RuntimeEventType.VALIDATION_STARTED: ValidationPayloadV1.schema_id,
}


def register_payload_schema(
    schema_cls: type[T],
    *,
    extension: bool = False,
) -> type[T]:
    """Register a payload model. Set ``extension=True`` for agent/app custom schemas."""
    schema_id = schema_cls.schema_id
    target = _EXTENSION_REGISTRY if extension else _PAYLOAD_REGISTRY
    if schema_id in _PAYLOAD_REGISTRY or schema_id in _EXTENSION_REGISTRY:
        existing = _PAYLOAD_REGISTRY.get(schema_id) or _EXTENSION_REGISTRY.get(schema_id)
        if existing is not schema_cls:
            raise RuntimeEventPayloadError(f"duplicate payload schema_id: {schema_id!r}")
        return schema_cls
    target[schema_id] = schema_cls
    return schema_cls


def get_payload_schema(schema_id: str) -> type[RuntimeEventPayload] | None:
    return _PAYLOAD_REGISTRY.get(schema_id) or _EXTENSION_REGISTRY.get(schema_id)


def list_registered_payload_schema_ids(*, include_extensions: bool = True) -> list[str]:
    ids = sorted(_PAYLOAD_REGISTRY.keys())
    if include_extensions:
        ids.extend(sorted(_EXTENSION_REGISTRY.keys()))
    return ids


def bootstrap_canonical_payload_registry() -> None:
    """Idempotent registration of platform payload families."""
    for schema_cls in CANONICAL_PAYLOAD_TYPES:
        register_payload_schema(schema_cls)


bootstrap_canonical_payload_registry()


def merge_payload_envelope(
    base: dict[str, Any],
    typed: RuntimeEventPayload,
    *,
    promote_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Merge a typed envelope into a legacy payload dict.

    ``promote_fields`` copies selected keys to the top level for ops filters
    (e.g. ``tool_name`` on TOOL_* events).
    """
    merged = dict(base)
    merged.update(typed.to_envelope())
    if promote_fields:
        merged.update(promote_fields)
    return merged


def runtime_event_with_payload(
    event: RuntimeEvent,
    typed: RuntimeEventPayload,
    *,
    promote_fields: dict[str, Any] | None = None,
) -> RuntimeEvent:
    return event.model_copy(
        update={"payload": merge_payload_envelope(event.payload, typed, promote_fields=promote_fields)}
    )


def validate_payload_envelope(payload: dict[str, Any]) -> RuntimeEventPayload | None:
    """
    Validate ``payload_schema_id`` + ``data`` when present.

    Returns parsed payload instance, or ``None`` when envelope keys are absent
    (legacy unstructured payloads).
    """
    schema_id = payload.get("payload_schema_id")
    if schema_id is None:
        return None
    if not isinstance(schema_id, str) or not schema_id.strip():
        raise RuntimeEventPayloadError("payload_schema_id must be a non-empty string")
    schema_cls = get_payload_schema(schema_id)
    if schema_cls is None:
        raise UnknownPayloadSchemaError(f"unknown payload_schema_id: {schema_id!r}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeEventPayloadError("typed payload envelope requires data dict")
    return schema_cls.model_validate(data)


def assert_runtime_event_payload(event: RuntimeEvent) -> None:
    """Validate typed envelope on a ``RuntimeEvent`` when ``payload_schema_id`` is set."""
    validate_payload_envelope(event.payload)
