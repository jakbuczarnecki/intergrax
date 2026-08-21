# © Artur Czarnecki. All rights reserved.

"""Unit tests for proof.parcel_events staffing schema verification."""

from __future__ import annotations

from typing import Any

import pytest

from platform_proofs.tools.iterative_sql_investigation.setup import (
    FORBIDDEN_STAFFING_COLUMNS,
    verify_staffing_columns_absent,
)

pytestmark = pytest.mark.unit


class _FakeStore:
    def __init__(self, columns: tuple[str, ...]) -> None:
        self._columns = columns

    def fetch_all(self, sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
        del sql, params
        return [{"column_name": name} for name in self._columns]


def test_verify_staffing_columns_absent_passes_canonical_columns() -> None:
    store = _FakeStore(
        (
            "parcel_id",
            "created_at",
            "region",
            "origin_hub",
            "destination_hub",
            "carrier",
            "service_type",
            "route_type",
            "distance_km",
            "weight_kg",
            "planned_hours",
            "actual_hours",
            "delayed",
            "weekday",
        )
    )
    assert verify_staffing_columns_absent(store) is True


def test_verify_staffing_columns_absent_fails_when_forbidden_present() -> None:
    forbidden = next(iter(FORBIDDEN_STAFFING_COLUMNS))
    store = _FakeStore(("parcel_id", "region", forbidden))
    assert verify_staffing_columns_absent(store) is False
