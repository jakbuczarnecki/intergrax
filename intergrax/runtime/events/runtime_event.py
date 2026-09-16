# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Compatibility import path for canonical ``RuntimeEvent`` (contract-owned)."""

from __future__ import annotations

from intergrax.contracts.runtime_event import (
    RuntimeEvent,
    RuntimeEventType,
    parse_runtime_event_payload,
    register_runtime_event_catalog_enricher,
)


def _apply_catalog_defaults(event: RuntimeEvent) -> bool:
    from intergrax.runtime.events.event_catalog import get_catalog_entry

    entry = get_catalog_entry(event.event_type)
    if entry is None:
        return False
    if not event.event_kind:
        event.event_kind = entry.default_event_kind
    if event.event_category is None:
        event.event_category = entry.category
    if not event.ops_hint:
        event.ops_hint = entry.ops_hint
    return True


register_runtime_event_catalog_enricher(_apply_catalog_defaults)

__all__ = [
    "RuntimeEvent",
    "RuntimeEventType",
    "parse_runtime_event_payload",
]
