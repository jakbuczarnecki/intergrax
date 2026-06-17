# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deprecated view over ``event_catalog`` — use ``event_catalog`` as SSOT (OBS-EVOL-9.1)."""

from __future__ import annotations

from intergrax.runtime.events.event_catalog import (
    EVENT_CATALOG,
    EVENT_OPS_FILTER_HINTS,
    EVENT_PHASE_COVERAGE,
    OpsFilterHint,
    list_unmapped_event_types,
    list_unmapped_ops_filter_hints,
    ops_filter_hint_for_event,
    phase_for_event,
)

__all__ = [
    "EVENT_CATALOG",
    "EVENT_OPS_FILTER_HINTS",
    "EVENT_PHASE_COVERAGE",
    "OpsFilterHint",
    "list_unmapped_event_types",
    "list_unmapped_ops_filter_hints",
    "ops_filter_hint_for_event",
    "phase_for_event",
]
