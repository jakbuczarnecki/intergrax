# © Artur Czarnecki. All rights reserved.

"""Unit tests for TOOLS-ITERATIVE-SQL-INVESTIGATION deterministic dataset."""

from __future__ import annotations

import pytest

from platform_proofs.tools.iterative_sql_investigation.dataset import (
    ANOMALY_SEGMENT,
    DEFAULT_ROW_COUNT,
    HIGH_VOLUME_HUB,
    PROOF_ROW_COUNT,
    generate_parcel_events,
    verify_dataset_invariants,
)

pytestmark = pytest.mark.unit


def test_dataset_is_deterministic_for_fixed_seed() -> None:
    first = generate_parcel_events(row_count=256, seed=42)
    second = generate_parcel_events(row_count=256, seed=42)
    assert first == second


def test_dataset_supports_small_and_proof_sizes() -> None:
    small = generate_parcel_events(row_count=128, seed=42)
    proof = generate_parcel_events(row_count=PROOF_ROW_COUNT, seed=42)
    assert len(small) == 128
    assert len(proof) == PROOF_ROW_COUNT


def test_dataset_semantic_invariants() -> None:
    rows = generate_parcel_events(row_count=DEFAULT_ROW_COUNT, seed=42)
    invariants = verify_dataset_invariants(rows)
    assert invariants == {
        "north_worse_than_non_north": True,
        "naive_hub_trap_is_high_volume": True,
        "true_anomaly_segment_identified": True,
        "global_weight_delay_correlation": True,
        "no_within_segment_weight_signal": True,
        "staffing_variables_absent": True,
    }


def test_anomaly_segment_is_planted() -> None:
    rows = generate_parcel_events(row_count=DEFAULT_ROW_COUNT, seed=42)
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
