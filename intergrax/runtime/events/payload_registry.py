# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime event payload schema registry (OBS-BUS-1)."""

from __future__ import annotations

from typing import Any, TypeVar

from intergrax.runtime.events.payloads import CANONICAL_PAYLOAD_TYPES, RuntimeEventPayload
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.runtime_event_payload_policy import (
    EVENT_TYPE_PREFERRED_SCHEMA,
    PayloadWriteMode,
    RuntimeEventTypeClassification,
    get_runtime_event_payload_policy,
)

__all__ = [
    "EVENT_TYPE_PREFERRED_SCHEMA",
    "RuntimeEventPayloadError",
    "UnknownPayloadSchemaError",
    "assert_canonical_production_runtime_event_payload",
    "assert_runtime_event_payload",
    "bootstrap_canonical_payload_registry",
    "get_payload_schema",
    "list_registered_payload_schema_ids",
    "merge_payload_envelope",
    "register_payload_schema",
    "runtime_event_with_payload",
    "validate_payload_envelope",
]

T = TypeVar("T", bound=RuntimeEventPayload)

_PAYLOAD_REGISTRY: dict[str, type[RuntimeEventPayload]] = {}
_EXTENSION_REGISTRY: dict[str, type[RuntimeEventPayload]] = {}


class RuntimeEventPayloadError(ValueError):
    """Raised when a runtime event payload envelope is invalid."""


class UnknownPayloadSchemaError(RuntimeEventPayloadError):
    """Raised when payload_schema_id is not registered."""


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


def _uses_spine_event_kind(event: RuntimeEvent) -> bool:
    kind = event.event_kind
    if not kind:
        return True
    return kind == event.event_type.value


def assert_canonical_production_runtime_event_payload(event: RuntimeEvent) -> None:
    """
    Enforce typed payload on canonical production write boundaries.

    Spine event types with a preferred schema must carry a matching envelope.
    Custom ``event_kind`` values require a registered typed envelope (fail-closed).
    """
    policy = get_runtime_event_payload_policy(event.event_type)
    if policy.classification != RuntimeEventTypeClassification.CANONICAL_PRODUCTION:
        return
    if not _uses_spine_event_kind(event):
        schema_id = event.payload.get("payload_schema_id")
        if schema_id is None:
            raise RuntimeEventPayloadError(
                f"extension event_kind {event.event_kind!r} requires registered typed payload envelope"
            )
        validate_payload_envelope(event.payload)
        return
    if policy.write_mode == PayloadWriteMode.EXTENSION_EVENT_KIND:
        schema_id = event.payload.get("payload_schema_id")
        if schema_id is None:
            raise RuntimeEventPayloadError(
                f"canonical event {event.event_type.value} requires extension typed payload envelope"
            )
        validate_payload_envelope(event.payload)
        return
    preferred = policy.schema_id
    if preferred is None:
        raise RuntimeEventPayloadError(
            f"canonical production event {event.event_type.value} missing payload policy schema_id"
        )
    schema_id = event.payload.get("payload_schema_id")
    if schema_id is None:
        raise RuntimeEventPayloadError(
            f"canonical production event {event.event_type.value} requires typed payload envelope"
        )
    if schema_id != preferred:
        raise RuntimeEventPayloadError(
            f"payload_schema_id mismatch for {event.event_type.value}: "
            f"expected {preferred!r}, got {schema_id!r}"
        )
    validate_payload_envelope(event.payload)
