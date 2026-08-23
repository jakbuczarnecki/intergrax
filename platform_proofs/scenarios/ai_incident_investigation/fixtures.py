# © Artur Czarnecki. All rights reserved.

"""Synthetic manufacturing fixture — observable surface vs private evaluator truth."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class HypothesisId(StrEnum):
    """Scenario-local hypothesis identifiers (H1/H2/H3)."""

    H1 = "H1"
    H2 = "H2"
    H3 = "H3"


INCIDENT_WINDOW_START = datetime(2026, 3, 10, 6, 0, 0)
INCIDENT_WINDOW_END = datetime(2026, 3, 12, 18, 0, 0)
LINE_ID = "line4"


@dataclass(frozen=True, slots=True)
class _PrivateFixtureTruth:
    """Evaluator-only ground truth — never model-visible."""

    initiating_factor_code: str
    station_id: str


@dataclass(frozen=True, slots=True)
class WorkloadObservation:
    line_id: str
    window_label: str
    order_volume_delta_pct: float
    admissible: bool


@dataclass(frozen=True, slots=True)
class ThroughputObservation:
    line_id: str
    window_label: str
    target_attainment_pct: float
    baseline_attainment_pct: float
    admissible: bool


@dataclass(frozen=True, slots=True)
class TelemetryObservation:
    station_id: str
    window_label: str
    signal_state: str
    complex_assembly_throughput_pct: float
    admissible: bool


@dataclass(frozen=True, slots=True)
class IncidentFixture:
    """Scenario fixture with strict observable / hidden separation."""

    private_truth: _PrivateFixtureTruth
    workload: WorkloadObservation
    throughput: ThroughputObservation
    telemetry: TelemetryObservation


def build_skeleton_fixture() -> IncidentFixture:
    """Minimal adversarial slice: workload↑ + throughput↓ trap; telemetry distinguishes H3."""
    private_truth = _PrivateFixtureTruth(
        initiating_factor_code="station_signal_degraded_pattern",
        station_id="complex_assembly_station",
    )
    return IncidentFixture(
        private_truth=private_truth,
        workload=WorkloadObservation(
            line_id=LINE_ID,
            window_label="incident_window",
            order_volume_delta_pct=22.0,
            admissible=True,
        ),
        throughput=ThroughputObservation(
            line_id=LINE_ID,
            window_label="incident_window",
            target_attainment_pct=78.0,
            baseline_attainment_pct=94.0,
            admissible=True,
        ),
        telemetry=TelemetryObservation(
            station_id="complex_assembly_station",
            window_label="incident_window",
            signal_state="intermittent_degraded",
            complex_assembly_throughput_pct=62.0,
            admissible=True,
        ),
    )


# Strings that must never appear in model-visible/tool-visible material.
FORBIDDEN_LEAK_MARKERS: frozenset[str] = frozenset(
    {
        "station_signal_degraded_pattern",
        "initiating_factor_code",
        "hidden_root_cause",
        "correct_answer",
        "expected_diagnosis",
        "fixture_hidden_truth",
    }
)
