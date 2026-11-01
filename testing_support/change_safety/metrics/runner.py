# © Artur Czarnecki. All rights reserved.

"""W6-01 metric specs and verdict arithmetic for EVOL-M-01..11 (canonical thresholds, OD-EVOL-03: no lowering).

Population measurement lives in the qualification layer (``tests/qualification/evol_x/metrics/measure.py``);
this module only fixes thresholds and decides PASS/FAIL so that no reading can be self-declared green.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MetricStatus(StrEnum):
    MEASURED = "MEASURED"
    NOT_MEASURED = "NOT_MEASURED"


class ThresholdKind(StrEnum):
    RATIO_MIN = "RATIO_MIN"  # numerator / denominator >= threshold
    COUNT_MAX = "COUNT_MAX"  # numerator <= threshold


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    kind: ThresholdKind
    threshold: float
    description: str


# Roadmap §3.0.9 (EVOL-M-01..11) as ratified canonical by OD-EVOL-03.
METRIC_SPECS: dict[str, MetricSpec] = {
    spec.metric_id: spec
    for spec in (
        MetricSpec("EVOL-M-01", ThresholdKind.RATIO_MIN, 0.95, "complete impact assessment of in-scope changes"),
        MetricSpec("EVOL-M-02", ThresholdKind.RATIO_MIN, 0.90, "contract/composition consumer edge coverage"),
        MetricSpec("EVOL-M-02C", ThresholdKind.COUNT_MAX, 0, "critical consumer omissions"),
        MetricSpec("EVOL-M-03", ThresholdKind.RATIO_MIN, 0.98, "selection precision (S1..S6 controlled set)"),
        MetricSpec("EVOL-M-03R", ThresholdKind.RATIO_MIN, 1.0, "selection recall vs independent oracle R*"),
        MetricSpec("EVOL-M-03C", ThresholdKind.RATIO_MIN, 1.0, "mandatory suite coverage"),
        MetricSpec("EVOL-M-03E", ThresholdKind.COUNT_MAX, 0, "relevant tests escaped from selection"),
        MetricSpec("EVOL-M-04", ThresholdKind.RATIO_MIN, 1.0, "no mandatory suite skipped when impact unknown"),
        MetricSpec("EVOL-M-05", ThresholdKind.RATIO_MIN, 1.0, "S6 injected regressions detected"),
        MetricSpec("EVOL-M-06", ThresholdKind.COUNT_MAX, 0, "undetected cross-application escapes"),
        MetricSpec("EVOL-M-07", ThresholdKind.RATIO_MIN, 1.0, "incompatible contract/schema arms detected (S2,S5)"),
        MetricSpec("EVOL-M-08", ThresholdKind.RATIO_MIN, 1.0, "qualified rollback or explicit refusal (S7)"),
        MetricSpec(
            "EVOL-M-09T",
            ThresholdKind.RATIO_MIN,
            1.0,
            "dossier drill tool: determinism on 3 seed paths, SLA <=120s",
        ),
        MetricSpec(
            "EVOL-M-09H",
            ThresholdKind.RATIO_MIN,
            1.0,
            "operator attribution drill (human); NOT_MEASURED until W7 supplement",
        ),
        MetricSpec(
            "EVOL-M-09A",
            ThresholdKind.RATIO_MIN,
            1.0,
            "automation-first scoped C1-C3 diagnostic qualification (OD-EVOL-13); NOT_MEASURED until FINAL scoped",
        ),
        MetricSpec("EVOL-M-10", ThresholdKind.COUNT_MAX, 0, "unauthorized platform-core edits undetected (S8)"),
        MetricSpec("EVOL-M-11", ThresholdKind.RATIO_MIN, 1.0, "requalification triggers executable (T1..T9)"),
    )
}

BASE_METRIC_IDS = tuple(f"EVOL-M-{i:02d}" for i in range(1, 12))


@dataclass(frozen=True)
class MetricReading:
    metric_id: str
    numerator: int
    denominator: int
    status: MetricStatus
    threshold: float | None = None
    passed: bool | None = None
    population_digest: str | None = None
    note: str = ""


def make_reading(
    metric_id: str,
    numerator: int,
    denominator: int,
    *,
    population_digest: str | None,
    note: str = "",
    extra_condition: bool = True,
) -> MetricReading:
    """Build a MEASURED reading; an empty population is NOT_MEASURED (never a vacuous PASS)."""
    spec = METRIC_SPECS[metric_id]
    if spec.kind is ThresholdKind.RATIO_MIN and denominator <= 0:
        return MetricReading(
            metric_id, numerator, denominator, MetricStatus.NOT_MEASURED, spec.threshold, None, None, note
        )
    if spec.kind is ThresholdKind.RATIO_MIN:
        passed = numerator / denominator >= spec.threshold
    else:
        passed = numerator <= spec.threshold
    return MetricReading(
        metric_id,
        numerator,
        denominator,
        MetricStatus.MEASURED,
        spec.threshold,
        bool(passed and extra_condition),
        population_digest,
        note,
    )
