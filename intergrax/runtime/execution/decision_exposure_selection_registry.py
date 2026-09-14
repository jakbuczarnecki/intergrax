# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Entry-point registry for ``DecisionExposureSelectionStrategy`` plugins."""

from __future__ import annotations

from intergrax.contracts.decision_exposure_selection import DecisionExposureSelectionStrategy
from intergrax.core.plugins.discovery import (
    EP_DECISION_EXPOSURE_SELECTION_STRATEGIES,
    get_entry_point_spec,
    instantiate_entry_point_target,
    iter_entry_point_specs,
    load_entry_point_value,
)

_ENTRY_POINT_GROUP = EP_DECISION_EXPOSURE_SELECTION_STRATEGIES


def load_decision_exposure_selection_strategy(
    strategy_id: str,
) -> DecisionExposureSelectionStrategy[object] | None:
    """Load a strategy by entry-point name."""
    spec = get_entry_point_spec(_ENTRY_POINT_GROUP, strategy_id)
    if spec is None:
        return None
    loaded = load_entry_point_value(spec.value)
    instance = instantiate_entry_point_target(loaded)
    if not isinstance(instance, DecisionExposureSelectionStrategy):
        raise TypeError(
            f"decision exposure selection entry point {spec.name!r} must return "
            "DecisionExposureSelectionStrategy",
        )
    return instance


def list_decision_exposure_selection_strategy_ids() -> tuple[str, ...]:
    """Return registered entry-point strategy ids (sorted)."""
    return tuple(spec.name for spec in iter_entry_point_specs(_ENTRY_POINT_GROUP))
