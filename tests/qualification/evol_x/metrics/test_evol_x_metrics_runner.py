# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.metrics.runner import MetricStatus, run_evol_metrics_snapshot

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_x_metrics_runner_emits_m01_m11() -> None:
    readings = run_evol_metrics_snapshot()
    ids = {r.metric_id for r in readings}
    assert ids == {f"EVOL-M-{i:02d}" for i in range(1, 12)}
    m01 = next(r for r in readings if r.metric_id == "EVOL-M-01")
    assert m01.status is MetricStatus.MEASURED
    m03 = next(r for r in readings if r.metric_id == "EVOL-M-03")
    assert m03.status is MetricStatus.NOT_MEASURED
