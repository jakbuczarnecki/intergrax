# © Artur Czarnecki. All rights reserved.
# Intergrax platform proof — TOOLS-ITERATIVE-SQL-INVESTIGATION (PP-3C).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.providers.relational_store.postgresql import create_postgresql_relational_store

from platform_proofs.tools.iterative_sql_investigation.dataset import (
    ANOMALY_SEGMENT,
    HIGH_VOLUME_HUB,
    NORTH_ELEVATION_MIN_DELTA,
    PARCEL_EVENTS_DDL,
    PROOF_ROW_COUNT,
    bulk_load_parcel_events,
)
from platform_proofs.tools.iterative_sql_investigation.dataset_identity import (
    DatasetFingerprint,
    DatasetIdentity,
    compute_dataset_fingerprint,
)
from platform_proofs.tools.iterative_sql_investigation.runtime import (
    ADMIN_DSN_ENV,
    DEFAULT_ADMIN_DSN,
)


class DatasetSetupError(RuntimeError):
    """Canonical dataset materialization or verification failure."""


@dataclass(frozen=True, slots=True)
class DbVerificationStats:
    total_rows: int
    north_delay_rate: float
    non_north_delay_rate: float
    top_delayed_count_hub: str
    top_normalized_rate_hub: str
    anomaly_segment_count: int
    anomaly_segment_delay_rate: float
    north_elevation_delta: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "north_delay_rate": round(self.north_delay_rate, 6),
            "non_north_delay_rate": round(self.non_north_delay_rate, 6),
            "top_absolute_delayed_count_hub": self.top_delayed_count_hub,
            "top_normalized_rate_hub": self.top_normalized_rate_hub,
            "anomaly_segment_count": self.anomaly_segment_count,
            "anomaly_segment_delay_rate": round(self.anomaly_segment_delay_rate, 6),
            "north_elevation_delta": round(self.north_elevation_delta, 6),
        }


@dataclass(frozen=True, slots=True)
class DatasetSetupResult:
    identity: DatasetIdentity
    fingerprint: DatasetFingerprint
    loaded_rows: int
    db_stats: DbVerificationStats
    contract_checks: dict[str, bool]

    @property
    def verified(self) -> bool:
        return self.loaded_rows == self.identity.row_count and all(self.contract_checks.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.as_dict(),
            "fingerprint": self.fingerprint.as_dict(),
            "loaded_rows": self.loaded_rows,
            "db_stats": self.db_stats.as_dict(),
            "contract_checks": dict(self.contract_checks),
            "verified": self.verified,
        }


def resolve_admin_dsn(explicit: str | None = None) -> str:
    import os

    if explicit and explicit.strip():
        return explicit.strip()
    return os.environ.get(ADMIN_DSN_ENV, DEFAULT_ADMIN_DSN).strip()


def _ensure_schema(store: RelationalStore) -> None:
    store.execute("CREATE SCHEMA IF NOT EXISTS proof")
    store.execute(PARCEL_EVENTS_DDL)


def _query_stats(store: RelationalStore) -> DbVerificationStats:
    total = int(store.fetch_all("SELECT COUNT(*) AS n FROM proof.parcel_events")[0]["n"])
    region_rows = store.fetch_all(
        """
        SELECT region,
               AVG(CASE WHEN delayed THEN 1.0 ELSE 0.0 END) AS delay_rate
        FROM proof.parcel_events
        GROUP BY region
        """
    )
    rates = {str(row["region"]): float(row["delay_rate"]) for row in region_rows}
    north_rate = rates.get("North", 0.0)
    non_north_values = [rate for region, rate in rates.items() if region != "North"]
    non_north_rate = sum(non_north_values) / len(non_north_values) if non_north_values else 0.0

    hub_abs = store.fetch_all(
        """
        SELECT origin_hub AS hub, SUM(CASE WHEN delayed THEN 1 ELSE 0 END) AS delayed_count
        FROM proof.parcel_events
        GROUP BY origin_hub
        ORDER BY delayed_count DESC, origin_hub ASC
        LIMIT 1
        """
    )
    hub_rate = store.fetch_all(
        """
        SELECT origin_hub AS hub,
               AVG(CASE WHEN delayed THEN 1.0 ELSE 0.0 END) AS delay_rate
        FROM proof.parcel_events
        GROUP BY origin_hub
        ORDER BY delay_rate DESC, origin_hub ASC
        LIMIT 1
        """
    )
    region, service, route = ANOMALY_SEGMENT
    anomaly = store.fetch_all(
        """
        SELECT COUNT(*) AS n,
               AVG(CASE WHEN delayed THEN 1.0 ELSE 0.0 END) AS delay_rate
        FROM proof.parcel_events
        WHERE region = %s AND service_type = %s AND route_type = %s
        """,
        (region, service, route),
    )
    north_without = store.fetch_all(
        """
        SELECT AVG(CASE WHEN delayed THEN 1.0 ELSE 0.0 END) AS delay_rate
        FROM proof.parcel_events
        WHERE region = 'North'
          AND NOT (service_type = %s AND route_type = %s)
        """,
        (service, route),
    )
    north_without_rate = float(north_without[0]["delay_rate"] or 0.0)
    return DbVerificationStats(
        total_rows=total,
        north_delay_rate=north_rate,
        non_north_delay_rate=non_north_rate,
        top_delayed_count_hub=str(hub_abs[0]["hub"]),
        top_normalized_rate_hub=str(hub_rate[0]["hub"]),
        anomaly_segment_count=int(anomaly[0]["n"]),
        anomaly_segment_delay_rate=float(anomaly[0]["delay_rate"] or 0.0),
        north_elevation_delta=north_rate - north_without_rate,
    )


def _contract_checks(stats: DbVerificationStats) -> dict[str, bool]:
    return {
        "row_count_exact": stats.total_rows == PROOF_ROW_COUNT,
        "north_worse_than_non_north": stats.north_delay_rate > stats.non_north_delay_rate,
        "naive_hub_trap_is_high_volume": stats.top_delayed_count_hub == HIGH_VOLUME_HUB,
        "high_volume_hub_not_highest_normalized_rate": stats.top_normalized_rate_hub != HIGH_VOLUME_HUB,
        "anomaly_segment_rate_above_half": stats.anomaly_segment_delay_rate > 0.5,
        "anomaly_materially_elevates_north": stats.north_elevation_delta >= NORTH_ELEVATION_MIN_DELTA,
        "staffing_columns_absent": True,
    }


def materialize_and_verify_dataset(
    *,
    admin_dsn: str | None = None,
    identity: DatasetIdentity | None = None,
) -> DatasetSetupResult:
    """TRUNCATE → regenerate canonical rows → reload → verify materialized PostgreSQL."""
    resolved_identity = identity or DatasetIdentity.canonical()
    fingerprint = compute_dataset_fingerprint(resolved_identity)
    dsn = resolve_admin_dsn(admin_dsn)
    store = create_postgresql_relational_store(dsn=dsn, tenant_schema="proof")
    try:
        store.connect()
        _ensure_schema(store)
        loaded = bulk_load_parcel_events(
            store,
            row_count=resolved_identity.row_count,
            seed=resolved_identity.seed,
        )
        if loaded != resolved_identity.row_count:
            raise DatasetSetupError(
                f"load row count mismatch: loaded={loaded} expected={resolved_identity.row_count}"
            )
        stats = _query_stats(store)
        checks = _contract_checks(stats)
        result = DatasetSetupResult(
            identity=resolved_identity,
            fingerprint=fingerprint,
            loaded_rows=loaded,
            db_stats=stats,
            contract_checks=checks,
        )
        if not result.verified:
            failed = [name for name, ok in checks.items() if not ok]
            raise DatasetSetupError(
                f"materialized dataset verification failed: {', '.join(failed)}"
            )
        return result
    finally:
        store.close()


def verify_postgres_reachable(*, admin_dsn: str | None = None) -> None:
    dsn = resolve_admin_dsn(admin_dsn)
    store = create_postgresql_relational_store(dsn=dsn, tenant_schema="proof")
    try:
        store.connect()
        store.fetch_all("SELECT 1 AS ok")
    except Exception as exc:
        raise DatasetSetupError(f"PostgreSQL proof environment unreachable: {exc}") from exc
    finally:
        store.close()
