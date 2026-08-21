# © Artur Czarnecki. All rights reserved.

"""Unit tests for TOOLS-ITERATIVE-SQL-INVESTIGATION deterministic dataset."""

from __future__ import annotations

import pytest

from platform_proofs.tools.iterative_sql_investigation.dataset import (
    ANOMALY_HUB,
    ANOMALY_SEGMENT,
    DEFAULT_ROW_COUNT,
    HIGH_VOLUME_HUB,
    PROOF_ROW_COUNT,
    generate_parcel_events,
    verify_dataset_invariants,
)

pytestmark = pytest.mark.unit

_EXPECTED_INVARIANTS = {
    "north_worse_than_non_north": True,
    "naive_hub_trap_is_high_volume": True,
    "high_volume_hub_not_highest_normalized_rate": True,
    "true_anomaly_segment_in_north": True,
    "true_anomaly_segment_identified": True,
    "anomaly_materially_elevates_north": True,
    "global_weight_delay_correlation": True,
    "no_within_segment_weight_signal": True,
    "staffing_variables_absent": True,
}


def test_dataset_is_deterministic_for_fixed_seed() -> None:
    first = generate_parcel_events(row_count=256, seed=42)
    second = generate_parcel_events(row_count=256, seed=42)
    assert first == second


def test_dataset_supports_small_and_proof_sizes() -> None:
    small = generate_parcel_events(row_count=128, seed=42)
    proof = generate_parcel_events(row_count=PROOF_ROW_COUNT, seed=42)
    assert len(small) == 128
    assert len(proof) == PROOF_ROW_COUNT


@pytest.mark.parametrize("row_count", [DEFAULT_ROW_COUNT, PROOF_ROW_COUNT])
def test_dataset_semantic_invariants(row_count: int) -> None:
    rows = generate_parcel_events(row_count=row_count, seed=42)
    invariants = verify_dataset_invariants(rows)
    assert invariants == _EXPECTED_INVARIANTS


def test_anomaly_segment_is_planted_in_north() -> None:
    rows = generate_parcel_events(row_count=DEFAULT_ROW_COUNT, seed=42)
    assert ANOMALY_SEGMENT[0] == "North"
    anomaly_rows = [
        row
        for row in rows
        if (row.region, row.service_type, row.route_type) == ANOMALY_SEGMENT
    ]
    assert anomaly_rows
    delayed = sum(1 for row in anomaly_rows if row.delayed)
    assert delayed / len(anomaly_rows) > 0.5


def test_naive_hub_trap_prefers_high_volume_hub() -> None:
    rows = generate_parcel_events(row_count=DEFAULT_ROW_COUNT, seed=42)
    delayed_counts: dict[str, int] = {}
    for row in rows:
        if row.delayed:
            delayed_counts[row.origin_hub] = delayed_counts.get(row.origin_hub, 0) + 1
    assert max(delayed_counts, key=delayed_counts.get) == HIGH_VOLUME_HUB


def test_normalized_hub_analysis_falsifies_naive_volume_conclusion() -> None:
    rows = generate_parcel_events(row_count=DEFAULT_ROW_COUNT, seed=42)
    hub_totals: dict[str, int] = {}
    hub_delayed: dict[str, int] = {}
    for row in rows:
        hub_totals[row.origin_hub] = hub_totals.get(row.origin_hub, 0) + 1
        if row.delayed:
            hub_delayed[row.origin_hub] = hub_delayed.get(row.origin_hub, 0) + 1
    hub_rates = {
        hub: hub_delayed.get(hub, 0) / total
        for hub, total in hub_totals.items()
        if total > 0
    }
    assert max(hub_rates, key=hub_rates.get) != HIGH_VOLUME_HUB
    assert max(hub_rates, key=hub_rates.get) == ANOMALY_HUB


def test_anomaly_materially_explains_north_elevation() -> None:
    rows = generate_parcel_events(row_count=DEFAULT_ROW_COUNT, seed=42)
    north_rows = [row for row in rows if row.region == "North"]
    north_without_anomaly = [
        row
        for row in north_rows
        if (row.region, row.service_type, row.route_type) != ANOMALY_SEGMENT
    ]
    north_rate = sum(1 for row in north_rows if row.delayed) / len(north_rows)
    north_without_anomaly_rate = sum(1 for row in north_without_anomaly if row.delayed) / len(
        north_without_anomaly
    )
    non_north_rate = sum(1 for row in rows if row.region != "North" and row.delayed) / sum(
        1 for row in rows if row.region != "North"
    )
    assert north_rate > non_north_rate
    assert north_rate - north_without_anomaly_rate >= 0.01
    assert north_without_anomaly_rate <= non_north_rate + 0.02
