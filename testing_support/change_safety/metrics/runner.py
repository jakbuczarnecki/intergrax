# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.graph.coverage import compute_graph_coverage
from testing_support.change_safety.intake.git_change_set import rev_parse_head


class MetricStatus(StrEnum):
    MEASURED = "MEASURED"
    NOT_MEASURED = "NOT_MEASURED"


@dataclass(frozen=True)
class MetricReading:
    metric_id: str
    numerator: int
    denominator: int
    status: MetricStatus
    population_digest: str | None = None


def run_evol_metrics_snapshot() -> tuple[MetricReading, ...]:
    head = rev_parse_head()
    inventory = build_closed_world_inventory(source_commit_sha=head)
    coverage = compute_graph_coverage(inventory)
    readings: list[MetricReading] = []
    for idx in range(1, 12):
        metric_id = f"EVOL-M-{idx:02d}"
        if metric_id == "EVOL-M-01":
            readings.append(
                MetricReading(
                    metric_id=metric_id,
                    numerator=1,
                    denominator=1,
                    status=MetricStatus.MEASURED,
                    population_digest=inventory.inventory_digest,
                )
            )
        elif metric_id == "EVOL-M-02":
            readings.append(
                MetricReading(
                    metric_id=metric_id,
                    numerator=coverage.modules_with_import_edges,
                    denominator=coverage.module_nodes,
                    status=MetricStatus.MEASURED if coverage.module_nodes else MetricStatus.NOT_MEASURED,
                    population_digest=inventory.inventory_digest,
                )
            )
        else:
            readings.append(
                MetricReading(
                    metric_id=metric_id,
                    numerator=0,
                    denominator=0,
                    status=MetricStatus.NOT_MEASURED,
                )
            )
    return tuple(readings)
