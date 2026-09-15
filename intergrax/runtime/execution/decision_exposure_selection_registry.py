# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Metadata-only discovery helpers for exposure selection entry points."""

from __future__ import annotations

from intergrax.core.plugins.discovery import (
    EP_DECISION_EXPOSURE_SELECTION_STRATEGIES,
    iter_entry_point_specs,
)

_ENTRY_POINT_GROUP = EP_DECISION_EXPOSURE_SELECTION_STRATEGIES


def list_decision_exposure_selection_strategy_ids() -> tuple[str, ...]:
    """Return registered entry-point names (metadata only, no import)."""
    return tuple(spec.name for spec in iter_entry_point_specs(_ENTRY_POINT_GROUP))
