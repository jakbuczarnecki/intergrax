# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.nexus.observability import planning_metrics

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_record_planner_latency_and_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(planning_metrics, "_planner_latency_ms_total", 0.0)
    monkeypatch.setattr(planning_metrics, "_planner_fallback_total", 0)
    monkeypatch.setattr(planning_metrics, "_planning_failure_counts", {})

    planning_metrics.record_planner_latency(latency_ms=12.5)
    planning_metrics.record_planner_fallback()
    planning_metrics.record_planning_failure(kind="classifier_fallback")

    exported = planning_metrics.export_planning_metrics()
    assert exported["ops_planning_latency_ms_total"] == 12.5
    assert exported["ops_planning_fallback_total"] == 1.0
    assert exported["ops_planning_failure_classifier_fallback_total"] == 1.0
