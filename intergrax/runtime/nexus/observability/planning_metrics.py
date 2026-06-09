# © Artur Czarnecki. All rights reserved.

"""Planning observability metrics (COG-OBS.1)."""

from __future__ import annotations

_planner_latency_ms_total = 0.0
_planner_fallback_total = 0
_planning_failure_counts: dict[str, int] = {}


def record_planner_latency(*, latency_ms: float) -> None:
    global _planner_latency_ms_total
    _planner_latency_ms_total += latency_ms


def record_planner_fallback() -> None:
    global _planner_fallback_total
    _planner_fallback_total += 1


def record_planning_failure(*, kind: str) -> None:
    normalized = kind.strip() or "unknown"
    _planning_failure_counts[normalized] = _planning_failure_counts.get(normalized, 0) + 1


def export_planning_metrics() -> dict[str, float]:
    metrics = {
        "ops_planning_latency_ms_total": _planner_latency_ms_total,
        "ops_planning_fallback_total": float(_planner_fallback_total),
    }
    metrics.update(
        {
            f"ops_planning_failure_{key}_total": float(count)
            for key, count in _planning_failure_counts.items()
        }
    )
    return metrics
