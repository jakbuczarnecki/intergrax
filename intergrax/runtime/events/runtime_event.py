# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Compatibility import path for canonical ``RuntimeEvent`` (contract-owned)."""

from __future__ import annotations

from intergrax.contracts.runtime_event import (
    RuntimeEvent,
    RuntimeEventType,
    parse_runtime_event_payload,
)

__all__ = [
    "RuntimeEvent",
    "RuntimeEventType",
    "parse_runtime_event_payload",
]
