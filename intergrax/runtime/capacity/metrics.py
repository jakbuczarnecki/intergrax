# © Artur Czarnecki. All rights reserved.

"""Capacity metrics export (ECP-OBS.1)."""

from __future__ import annotations

_harness_scale_actions_total = 0
_replica_gauge: dict[str, int] = {}


def record_scale_action(*, target: str) -> None:
    global _harness_scale_actions_total
    _harness_scale_actions_total += 1
    _replica_gauge[target] = _replica_gauge.get(target, 0) + 1


def export_capacity_metrics() -> dict[str, float]:
    return {
        "harness_scale_actions_total": float(_harness_scale_actions_total),
        **{f"harness_replica_count_{k}": float(v) for k, v in _replica_gauge.items()},
    }
