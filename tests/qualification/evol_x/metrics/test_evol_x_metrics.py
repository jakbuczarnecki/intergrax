# © Artur Czarnecki. All rights reserved.

"""W6-01: EVOL-M-01..11 measured on real populations against canonical (OD-EVOL-03, no lowering) thresholds."""

import pytest
from testing_support.change_safety.metrics.runner import (
    BASE_METRIC_IDS,
    METRIC_SPECS,
    MetricStatus,
    ThresholdKind,
    make_reading,
)
from tests.qualification.evol_x.metrics.measure import collect_all

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_CANONICAL = {
    "EVOL-M-01": (ThresholdKind.RATIO_MIN, 0.95),
    "EVOL-M-02": (ThresholdKind.RATIO_MIN, 0.90),
    "EVOL-M-03": (ThresholdKind.RATIO_MIN, 0.98),
    "EVOL-M-04": (ThresholdKind.RATIO_MIN, 1.0),
    "EVOL-M-05": (ThresholdKind.RATIO_MIN, 1.0),
    "EVOL-M-06": (ThresholdKind.COUNT_MAX, 0),
    "EVOL-M-07": (ThresholdKind.RATIO_MIN, 1.0),
    "EVOL-M-08": (ThresholdKind.RATIO_MIN, 1.0),
    "EVOL-M-10": (ThresholdKind.COUNT_MAX, 0),
    "EVOL-M-11": (ThresholdKind.RATIO_MIN, 1.0),
}


@pytest.fixture(scope="module")
def readings(evol_inventory, evol_head):  # noqa: ANN001, ANN201
    return {r.metric_id: r for r in collect_all(evol_inventory, evol_head)}


def test_thresholds_match_ratified_canonical_values() -> None:
    for metric_id, (kind, value) in _CANONICAL.items():
        spec = METRIC_SPECS[metric_id]
        assert (spec.kind, spec.threshold) == (kind, value), metric_id


def test_every_base_metric_is_covered_by_a_reading(readings) -> None:  # noqa: ANN001
    covered = {mid.split("-")[0] + "-" + mid.split("-")[1] + "-" + mid.split("-")[2][:2] for mid in readings}
    assert set(BASE_METRIC_IDS) <= covered


@pytest.mark.parametrize("metric_id", sorted(METRIC_SPECS))
def test_metric_is_measured_on_a_real_population_and_meets_threshold(metric_id: str, readings) -> None:  # noqa: ANN001
    reading = readings[metric_id]
    assert reading.status is MetricStatus.MEASURED, f"{metric_id} NOT_MEASURED"
    assert reading.population_digest, f"{metric_id} has no population digest"
    assert reading.passed, (
        f"{metric_id}: {reading.numerator}/{reading.denominator} threshold={reading.threshold} {reading.note}"
    )


def test_populations_are_not_stubs(readings) -> None:  # noqa: ANN001
    assert readings["EVOL-M-01"].denominator >= 20  # P-H window
    assert readings["EVOL-M-02"].denominator >= 50  # real contract consumer pairs
    assert readings["EVOL-M-03"].denominator > 1 and readings["EVOL-M-03R"].denominator > 1


def test_make_reading_negative_cases() -> None:
    assert make_reading("EVOL-M-01", 0, 0, population_digest=None).status is MetricStatus.NOT_MEASURED
    assert make_reading("EVOL-M-01", 94, 100, population_digest="d").passed is False  # 94% < 95%: no lowering
    assert make_reading("EVOL-M-01", 95, 100, population_digest="d").passed is True
    assert make_reading("EVOL-M-06", 1, 5, population_digest="d").passed is False
    assert make_reading("EVOL-M-09", 3, 3, population_digest="d", extra_condition=False).passed is False
