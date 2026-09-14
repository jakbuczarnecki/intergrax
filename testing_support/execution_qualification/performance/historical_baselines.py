# © Artur Czarnecki. All rights reserved.

"""Documented historical legacy wall-time baselines (explicit provenance)."""

from __future__ import annotations

from testing_support.execution_qualification.catalog.profile_builders import (
    NPSC5E_R3_PROFILE_ID,
)
from testing_support.execution_qualification.performance.models import (
    PerformanceWallTimeProvenance,
    TimedWallSeconds,
)

# R3 performance record @ max_parallel=2 (pre-canonical coordinator matrix).
_NPSC5E_R3_HISTORICAL_WALL_SECONDS = 892.31

_HISTORICAL_BY_PROFILE: dict[str, TimedWallSeconds] = {
    NPSC5E_R3_PROFILE_ID: TimedWallSeconds(
        seconds=_NPSC5E_R3_HISTORICAL_WALL_SECONDS,
        provenance=PerformanceWallTimeProvenance.HISTORICAL_MEASURED,
        source_note=(
            "INTEGRAX_QUALIFICATION_EXECUTION_GRAPH_AUDIT R3 record; "
            "max_parallel=2; legacy flat coordinator mandatory matrix"
        ),
    ),
}


def historical_legacy_wall_for_profile(profile_id: str) -> TimedWallSeconds:
    baseline = _HISTORICAL_BY_PROFILE.get(profile_id)
    if baseline is not None:
        return baseline
    return TimedWallSeconds(
        seconds=None,
        provenance=PerformanceWallTimeProvenance.NOT_AVAILABLE,
        source_note="no verified historical full legacy wall time for equivalent scope",
    )
