# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Extension ``event_kind`` registry for Tier-2/3 domain signals (OBS-EVOL-9.4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.events.event_kind import DomainSignalError, validate_event_kind
from intergrax.runtime.events.event_taxonomy import category_for_event_kind
from intergrax.runtime.events.payload_registry import get_payload_schema

_HOS_STREAM_PREFIX = "intergrax.llm.stream."


@dataclass(frozen=True, slots=True)
class EventKindRegistryEntry:
    kind: str
    payload_schema_id: str
    category: str


_REGISTRY: dict[str, EventKindRegistryEntry] = {}


class EventKindRegistryError(DomainSignalError):
    """Raised when an extension event kind violates registry rules."""


def register_event_kind(
    kind: str,
    payload_schema_id: str,
    *,
    replace: bool = False,
) -> EventKindRegistryEntry:
    """Register a domain ``event_kind`` bound to an extension payload schema."""
    validate_event_kind(kind)
    if kind.startswith(_HOS_STREAM_PREFIX):
        raise EventKindRegistryError(
            f"event_kind {kind!r} is reserved for LLM stream chunks; use agents.* or applications.*"
        )
    if get_payload_schema(payload_schema_id) is None:
        raise EventKindRegistryError(
            f"payload_schema_id {payload_schema_id!r} is not registered"
        )
    existing = _REGISTRY.get(kind)
    if existing is not None and not replace:
        if existing.payload_schema_id != payload_schema_id:
            raise EventKindRegistryError(f"duplicate event_kind with different schema: {kind!r}")
        return existing
    entry = EventKindRegistryEntry(
        kind=kind,
        payload_schema_id=payload_schema_id,
        category=category_for_event_kind(kind).value,
    )
    _REGISTRY[kind] = entry
    return entry


def get_event_kind_entry(kind: str) -> EventKindRegistryEntry | None:
    return _REGISTRY.get(kind)


def require_registered_event_kind(kind: str) -> EventKindRegistryEntry:
    validate_event_kind(kind)
    entry = _REGISTRY.get(kind)
    if entry is None:
        raise EventKindRegistryError(
            f"unregistered event_kind: {kind!r} — call register_event_kind() at bootstrap"
        )
    return entry


def list_registered_event_kinds() -> list[str]:
    return sorted(_REGISTRY.keys())


def clear_event_kind_registry() -> None:
    """Test helper — clears extension kind registrations."""
    _REGISTRY.clear()
