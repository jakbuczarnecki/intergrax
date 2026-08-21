# © Artur Czarnecki. All rights reserved.
# Intergrax platform proof — TOOLS-ITERATIVE-SQL-INVESTIGATION (PP-3B).

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from intergrax.integrations.contracts.relational_store import RelationalStore

DEFAULT_SEED = 42
DEFAULT_ROW_COUNT = 5_000
PROOF_ROW_COUNT = 100_000

REGIONS: tuple[str, ...] = ("North", "South", "East", "West")
SERVICE_TYPES: tuple[str, ...] = ("standard", "express", "economy")
ROUTE_TYPES: tuple[str, ...] = ("local", "regional", "long_haul")
CARRIERS: tuple[str, ...] = ("CarrierA", "CarrierB", "CarrierC")

# Planted structure (documented for scenarios A/B/C):
# A — North aggregate delay rate is elevated by a North segment; North-Volume hub dominates naive
#     delayed counts but not normalized hub rates; North + express + long_haul is the true segment.
# B — Weight correlates with delay globally via route/service confounding, not within segments.
# C — No staffing variables exist in the schema.
ANOMALY_SEGMENT = ("North", "express", "long_haul")
ANOMALY_PARCEL_MODULUS = 31
HIGH_VOLUME_HUB = "North-Volume"
ANOMALY_HUB = "North-Hub"
TRUE_ANOMALY_RATE = 0.68
OTHER_REGION_RATE = 0.09
NORTH_ELEVATION_MIN_DELTA = 0.01

PARCEL_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS proof.parcel_events (
    parcel_id BIGINT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    region TEXT NOT NULL,
    origin_hub TEXT NOT NULL,
    destination_hub TEXT NOT NULL,
    carrier TEXT NOT NULL,
    service_type TEXT NOT NULL,
    route_type TEXT NOT NULL,
    distance_km NUMERIC(8, 2) NOT NULL,
    weight_kg NUMERIC(8, 2) NOT NULL,
    planned_hours NUMERIC(8, 2) NOT NULL,
    actual_hours NUMERIC(8, 2) NOT NULL,
    delayed BOOLEAN NOT NULL,
    weekday SMALLINT NOT NULL
)
"""


@dataclass(frozen=True, slots=True)
class ParcelEventRow:
    parcel_id: int
    created_at: datetime
    region: str
    origin_hub: str
    destination_hub: str
    carrier: str
    service_type: str
    route_type: str
    distance_km: float
    weight_kg: float
    planned_hours: float
    actual_hours: float
    delayed: bool
    weekday: int

    def as_insert_params(self) -> tuple[Any, ...]:
        return (
            self.parcel_id,
            self.created_at,
            self.region,
            self.origin_hub,
            self.destination_hub,
            self.carrier,
            self.service_type,
            self.route_type,
            self.distance_km,
            self.weight_kg,
            self.planned_hours,
            self.actual_hours,
            self.delayed,
            self.weekday,
        )


def _deterministic_unit(seed: int, parcel_id: int) -> float:
    mixed = (seed * 1_000_003) ^ (parcel_id * 2654435761)
    return ((mixed & 0xFFFFFFFF) / 0xFFFFFFFF)


def _pick(values: Sequence[str], seed: int, parcel_id: int, salt: int) -> str:
    index = int(_deterministic_unit(seed + salt, parcel_id) * len(values))
    return values[min(index, len(values) - 1)]


def _region_for(parcel_id: int, seed: int) -> str:
    unit = _deterministic_unit(seed, parcel_id)
    if unit < 0.40:
        return "North"
    if unit < 0.65:
        return "South"
    if unit < 0.85:
        return "East"
    return "West"


def _route_and_service(region: str, parcel_id: int, seed: int) -> tuple[str, str]:
    if region == ANOMALY_SEGMENT[0] and parcel_id % ANOMALY_PARCEL_MODULUS == 0:
        return ANOMALY_SEGMENT[2], ANOMALY_SEGMENT[1]
    if region == "North":
        route_type = "long_haul" if parcel_id % 2 == 0 else "regional"
    else:
        route_type = _pick(("local", "regional"), seed, parcel_id, 11)
    service_type = _pick(SERVICE_TYPES, seed, parcel_id, 17)
    if route_type == "long_haul" and service_type == "express":
        service_type = "standard"
    return route_type, service_type


def _weight_kg(route_type: str, service_type: str, parcel_id: int, seed: int) -> float:
    base = {"local": 4.0, "regional": 12.0, "long_haul": 28.0}[route_type]
    service_boost = {"economy": 0.0, "standard": 3.0, "express": 6.0}[service_type]
    jitter = _deterministic_unit(seed + 29, parcel_id) * 8.0
    return round(base + service_boost + jitter, 2)


def _segment_delay_probability(region: str, service_type: str, route_type: str) -> float:
    if (region, service_type, route_type) == ANOMALY_SEGMENT:
        return TRUE_ANOMALY_RATE
    return OTHER_REGION_RATE


def _delay_roll_unit(seed: int, parcel_id: int, region: str) -> float:
    mixed_parcel = parcel_id * 31 + sum(map(ord, region))
    return _deterministic_unit(seed + 53, mixed_parcel)


def _is_delayed(region: str, route_type: str, service_type: str, parcel_id: int, seed: int) -> bool:
    probability = _segment_delay_probability(region, service_type, route_type)
    return _delay_roll_unit(seed, parcel_id, region) < probability


def generate_parcel_events(*, row_count: int = DEFAULT_ROW_COUNT, seed: int = DEFAULT_SEED) -> list[ParcelEventRow]:
    if row_count < 1:
        raise ValueError("row_count must be >= 1")
    base_time = datetime(2025, 1, 6, 8, 0, tzinfo=timezone.utc)
    rows: list[ParcelEventRow] = []
    for parcel_id in range(1, row_count + 1):
        region = _region_for(parcel_id, seed)
        route_type, service_type = _route_and_service(region, parcel_id, seed)
        if (region, service_type, route_type) == ANOMALY_SEGMENT:
            origin_hub = ANOMALY_HUB
        elif region == "North" and parcel_id % 3 != 0:
            origin_hub = HIGH_VOLUME_HUB
        else:
            origin_hub = f"{region}-Hub"
        destination_hub = f"{_pick(REGIONS, seed, parcel_id, 23)}-Dest"
        weight = _weight_kg(route_type, service_type, parcel_id, seed)
        distance = {"local": 35.0, "regional": 180.0, "long_haul": 620.0}[route_type]
        distance += _deterministic_unit(seed + 31, parcel_id) * 25.0
        planned = max(1.0, distance / 80.0)
        delayed = _is_delayed(region, route_type, service_type, parcel_id, seed)
        actual = planned * (1.35 if delayed else 1.02)
        created_at = base_time + timedelta(hours=parcel_id % 168)
        rows.append(
            ParcelEventRow(
                parcel_id=parcel_id,
                created_at=created_at,
                region=region,
                origin_hub=origin_hub,
                destination_hub=destination_hub,
                carrier=_pick(CARRIERS, seed, parcel_id, 37),
                service_type=service_type,
                route_type=route_type,
                distance_km=round(distance, 2),
                weight_kg=weight,
                planned_hours=round(planned, 2),
                actual_hours=round(actual, 2),
                delayed=delayed,
                weekday=created_at.weekday(),
            )
        )
    return rows


def iter_parcel_events(*, row_count: int = DEFAULT_ROW_COUNT, seed: int = DEFAULT_SEED) -> Iterator[ParcelEventRow]:
    yield from generate_parcel_events(row_count=row_count, seed=seed)


def _delay_rate(rows: Sequence[ParcelEventRow], *, predicate) -> float:
    filtered = [row for row in rows if predicate(row)]
    if not filtered:
        return 0.0
    return sum(1 for row in filtered if row.delayed) / len(filtered)


def _hub_delay_rates(rows: Sequence[ParcelEventRow]) -> dict[str, float]:
    totals: dict[str, int] = {}
    delayed: dict[str, int] = {}
    for row in rows:
        totals[row.origin_hub] = totals.get(row.origin_hub, 0) + 1
        if row.delayed:
            delayed[row.origin_hub] = delayed.get(row.origin_hub, 0) + 1
    return {
        hub: delayed.get(hub, 0) / totals[hub]
        for hub in totals
        if totals[hub] > 0
    }


def verify_dataset_invariants(rows: Sequence[ParcelEventRow]) -> dict[str, bool]:
    north_rate = _delay_rate(rows, predicate=lambda row: row.region == "North")
    non_north_rate = _delay_rate(rows, predicate=lambda row: row.region != "North")
    north_rate_excluding_anomaly = _delay_rate(
        rows,
        predicate=lambda row: row.region == "North"
        and (row.region, row.service_type, row.route_type) != ANOMALY_SEGMENT,
    )
    anomaly_rate = _delay_rate(
        rows,
        predicate=lambda row: (row.region, row.service_type, row.route_type) == ANOMALY_SEGMENT,
    )
    delayed_by_hub: dict[str, int] = {}
    for row in rows:
        if row.delayed:
            delayed_by_hub[row.origin_hub] = delayed_by_hub.get(row.origin_hub, 0) + 1
    naive_top_hub = max(delayed_by_hub, key=delayed_by_hub.get)
    hub_rates = _hub_delay_rates(rows)
    normalized_top_hub = max(hub_rates, key=hub_rates.get)
    segment_rates: dict[tuple[str, str, str], list[bool]] = {}
    for row in rows:
        key = (row.region, row.service_type, row.route_type)
        segment_rates.setdefault(key, []).append(row.delayed)
    top_segment = max(
        segment_rates,
        key=lambda key: sum(segment_rates[key]) / len(segment_rates[key]),
    )
    heavy = [row for row in rows if row.weight_kg >= 20.0]
    light = [row for row in rows if row.weight_kg < 10.0]
    global_heavy_rate = _delay_rate(heavy, predicate=lambda row: True) if heavy else 0.0
    global_light_rate = _delay_rate(light, predicate=lambda row: True) if light else 0.0
    within_segment_weight_signal = False
    for key, delays in segment_rates.items():
        segment_rows = [row for row in rows if (row.region, row.service_type, row.route_type) == key]
        if len(segment_rows) < 30:
            continue
        heavy_rows = [row for row in segment_rows if row.weight_kg >= 20.0]
        light_rows = [row for row in segment_rows if row.weight_kg < 10.0]
        if not heavy_rows or not light_rows:
            continue
        heavy_rate = sum(1 for row in heavy_rows if row.delayed) / len(heavy_rows)
        light_rate = sum(1 for row in light_rows if row.delayed) / len(light_rows)
        if abs(heavy_rate - light_rate) > 0.08:
            within_segment_weight_signal = True
            break
    columns = {
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
    }
    staffing_absent = "staffing" not in columns and "staff_count" not in columns
    return {
        "north_worse_than_non_north": north_rate > non_north_rate,
        "naive_hub_trap_is_high_volume": naive_top_hub == HIGH_VOLUME_HUB,
        "high_volume_hub_not_highest_normalized_rate": normalized_top_hub != HIGH_VOLUME_HUB,
        "true_anomaly_segment_in_north": ANOMALY_SEGMENT[0] == "North",
        "true_anomaly_segment_identified": top_segment == ANOMALY_SEGMENT and anomaly_rate > 0.5,
        "anomaly_materially_elevates_north": (
            north_rate - north_rate_excluding_anomaly >= NORTH_ELEVATION_MIN_DELTA
        ),
        "global_weight_delay_correlation": global_heavy_rate > global_light_rate,
        "no_within_segment_weight_signal": not within_segment_weight_signal,
        "staffing_variables_absent": staffing_absent,
    }


INSERT_SQL = """
INSERT INTO proof.parcel_events (
    parcel_id, created_at, region, origin_hub, destination_hub, carrier,
    service_type, route_type, distance_km, weight_kg, planned_hours,
    actual_hours, delayed, weekday
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
"""


def bulk_load_parcel_events(
    store: RelationalStore,
    *,
    row_count: int = DEFAULT_ROW_COUNT,
    seed: int = DEFAULT_SEED,
    batch_size: int = 1_000,
) -> int:
    store.execute("TRUNCATE proof.parcel_events")
    loaded = 0
    batch: list[tuple[Any, ...]] = []
    for row in iter_parcel_events(row_count=row_count, seed=seed):
        batch.append(row.as_insert_params())
        if len(batch) >= batch_size:
            _execute_batch(store, batch)
            loaded += len(batch)
            batch.clear()
    if batch:
        _execute_batch(store, batch)
        loaded += len(batch)
    return loaded


def _execute_batch(store: RelationalStore, batch: Sequence[tuple[Any, ...]]) -> None:
    placeholders = ", ".join(
        f"({', '.join('%s' for _ in range(14))})" for _ in batch
    )
    flat: list[Any] = []
    for params in batch:
        flat.extend(params)
    sql = (
        "INSERT INTO proof.parcel_events ("
        "parcel_id, created_at, region, origin_hub, destination_hub, carrier, "
        "service_type, route_type, distance_km, weight_kg, planned_hours, "
        "actual_hours, delayed, weekday"
        f") VALUES {placeholders}"
    )
    store.execute(sql, flat)
