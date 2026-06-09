# © Artur Czarnecki. All rights reserved.

"""Planning observability metrics (COG-OBS.1)."""

from __future__ import annotations

_planner_latency_ms_total = 0.0
_planner_fallback_total = 0


def record_planner_latency(*, latency_ms: float) -> None:
    global _planner_latency_ms_total
    _planner_latency_ms_total += latency_ms


def record_planner_fallback() -> None:
    global _planner_fallback_total
    _planner_fallback_total += 1


def export_planning_metrics() -> dict[str, float]:
    return {
        "ops_planning_latency_ms_total": _planner_latency_ms_total,
        "ops_planning_fallback_total": float(_planner_fallback_total),
    }
